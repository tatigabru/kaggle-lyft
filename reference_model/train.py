import argparse
import collections
import pickle
import random
import pandas as pd
#from skimage import transforms
from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm, tqdm_notebook
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R
#from logger import Logger
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.
import os
os.environ["OMP_NUM_THREADS"] = "1"

# lyft SDK imports
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

# current project imports 
from unet import UNet
from BEVDataset import BEVImageDataset
from configs import OUTPUT_ROOT, IMG_SIZE, NUM_CLASSES
from transforms import (train_transform, test_transform, tensor_transform,
                        crop_d4_transforms, albu_valid_tansforms)



def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_unet_model(in_channels=3, num_output_classes=2):
    model = UNet(in_channels=in_channels, n_classes=num_output_classes, wf=6, depth=4, padding=True, up_mode='upsample')    
    # Optional, for multi GPU training and inference
    #model = nn.DataParallel(model)
    return model


def load_model_optim(model, optimizer, checkpoint_path: str):
    """Loads model weigths, optimizer, epoch, step abd loss to continuer training
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    print('Loaded model from {}, epoch {}'.format(checkpoint_path, epoch))
 
 
def train(model, model_name: str, data_folder: str, fold: int, debug=False, img_size=IMG_SIZE,
          epochs=15, batch_size = 8, num_workers=4, resume_weights='', resume_epoch=0):
    """
    Model training
    
    Input: 
        model : PyTorch model
        model_name : string name for model for checkpoints saving
        fold: evaluation fold number, 0-3
        debug: if True, runs the debugging on few images 
        img_size: size of images for training (for pregressive learning)
        epochs: number of epochs to train
        batch_size: number of images in batch
        num_workers: number of workers available
        resume_weights: directory with weights to resume (if avaialable)
        resume_epoch: number of epoch to continue training    
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # We weigh the loss for the 0 class lower to account for (some of) the big class imbalance
    class_weights = torch.from_numpy(np.array([0.2] + [1.0]*NUM_CLASSES, dtype=np.float32))
    class_weights = class_weights.to(device)

    #creates directories for checkpoints, tensorboard and predicitons
    checkpoints_dir = f'{OUTPUT_ROOT}/checkpoints/{model_name}_fold_{fold}'
    history_dir = f'{OUTPUT_ROOT}/history/{model_name}_fold_{fold}'
    predictions_dir = f'{OUTPUT_ROOT}/oof/{model_name}_fold_{fold}'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    print('\n', model_name, '\n')

    # choose inputs/targets
    input_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_input.png")))
    sample_tokens = [x.split("/")[-1].replace("_input.png","") for x in input_filepaths]
    sample_tokens = [x.replace("bev_data\\","") for x in sample_tokens]    

    # train samples
    df = pd.read_csv('train_samples.csv')
    train_df = df[df['samples'].isin(sample_tokens)]
    print('train samples: ', train_df.head())

    # validation samples
    df = pd.read_csv('val_samples.csv')
    valid_df = df[df['samples'].isin(sample_tokens)]
    print('valid samples: ', valid_df.head())

    # model
    # load weights to continue training
    if resume_weights != '':
        print('Load model from: {}'.format(resume_weights))
        checkpoint = torch.load(resume_weights)
        model.load_state_dict(checkpoint['model'])
        resume_epoch = checkpoint['epoch']+1
        	
    model = model.to(device)  

    # optimizer and schedulers
    learning_rate = 1e-4
    print(learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.2)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
  
    # datasets for train and validation
    train_dataset = BEVImageDataset(fold=fold, df=train_df, 
                                    debug=debug, img_size=img_size, 
                                    input_dir=data_folder,
                                    transforms = crop_d4_transforms)
    
    valid_dataset = BEVImageDataset(fold=fold, df=valid_df, 
                                    debug=debug, img_size=img_size, 
                                    input_dir=data_folder,
                                    transforms = None)                                

    # dataloaders for train and validation
    dataloader_train = DataLoader(train_dataset, 
                                  num_workers=num_workers,
                                  batch_size=batch_size,
                                  shuffle=True)
    
    dataloader_valid = DataLoader(valid_dataset,
                                  num_workers=num_workers,
                                  batch_size=4,
                                  shuffle=False)

    print('{} training images, {} validation images'.format(len(train_dataset), len(valid_dataset)))
    
    # training cycle
    print("Start training")
    all_losses, valid_losses = [], []
    
    for epoch in range(resume_epoch, epochs+1):
        print("Epoch", epoch)        
        epoch_losses = []
        progress_bar = tqdm(dataloader_train, total=len(dataloader_train))

        #with torch.set_grad_enabled(True):
        for iter_num, (img, target, sample_ids) in enumerate(progress_bar):
            img = img.to(device)  # [N, 3, H, W]
            target = target.to(device)  # [N, H, W] with class indices (0, 1)
            prediction = model(img)  # [N, 2, H, W]
            loss = F.cross_entropy(prediction, target, weight=class_weights)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 
            optimizer.step()                       
            epoch_losses.append(loss.detach().cpu().numpy())
            
            if iter_num == 0:
                visualize_predictions(img, prediction, target, predictions_dir, model_name, epoch)

        # loss history
        print("Epoch {}, Train Loss: {}".format(epoch, np.mean(epoch_losses)))
        all_losses.append(np.mean(epoch_losses))
        
        # validate model afterevery epoch
        valid_loss = validate(model, dataloader_valid, class_weights,
                              epoch, predictions_dir, save_oof = True)
        valid_losses.append(valid_loss)

        # save model, optimizer and scheduler after every epoch
        checkpoint_filename = "{}_fold_{}_epoch_{}.pth".format(model_name, fold, epoch)
        checkpoint_filepath = os.path.join(checkpoints_dir, checkpoint_filename)
        torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        #'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'loss': np.mean(epoch_losses),
        'valid_loss': valid_loss,
         }, checkpoint_filepath)  

    #plt.figure(figsize=(12,12))
    #plt.plot(all_losses, alpha=0.75, color = 'b')
    #plt.plot(valid_losses, alpha=0.75, color = 'r')
    #plt.savefig(os.path.join(history_dir, "training_history.png"))


def validate(model, dataloader_valid, class_weights, epoch: int, 
             predictions_dir: str, save_oof=True):
    """
    Validate model at the epoch end 
       
    Input: 
        model: current model 
        dataloader_valid: dataloader for the validation fold
        device: CUDA or CPU
        epoch: current epoch
        save_oof: boolean flag, if calculate oof predictions and save them in pickle 
        save_oof_numpy: boolean flag, if save oof predictions in numpy 
        predictions_dir: directory fro saving predictions
    Output:
        loss_valid: total validation loss, history 
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    with torch.no_grad():
        model.eval()
        val_losses = []
        progress_bar = tqdm(dataloader_valid, total=len(dataloader_valid))

        for iter_num, (img, target, sample_ids) in enumerate(progress_bar):
            img = img.to(device)  # [N, 3, H, W]
            target = target.to(device)  # [N, H, W] with class indices (0, 1)
            prediction = model(img)  # [N, 2, H, W]
            loss = F.cross_entropy(prediction, target, weight=class_weights)
            val_losses.append(loss.detach().cpu().numpy())
                  
    print("Epoch {}, Valid Loss: {}".format(epoch, np.mean(val_losses)))
        
    return np.mean(val_losses)    
        

def predict_test(model, dataset_test, device):    
    img, _, _ = dataset_test[0]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])       

    return prediction 


def visualize_predictions(input_image, prediction, target, predictions_dir: str, model_name: str, epoch: int, n_images=2, apply_softmax=True):
    """
    Takes as input 3 PyTorch tensors, plots the input image, predictions and targets
    The top images have two color channels: red for predictions, green for targets. 
        Note that red+green=yellow. In other words:
        Black: True Negative
        Green: False Negative
        Yellow: True Positive
        Red: False Positive
    Input: 
        input_image: the input image
        predictions: the predictions thresholded at 0.5 probability.
    """
    # Only select the first n images
    prediction = prediction[:n_images]
    target = target[:n_images]
    input_image = input_image[:n_images]

    prediction = prediction.detach().cpu().numpy()
    if apply_softmax:
        prediction = scipy.special.softmax(prediction, axis=1)
    class_one_preds = np.hstack(1-prediction[:,0])

    target = np.hstack(target.detach().cpu().numpy())

    class_rgb = np.repeat(class_one_preds[..., None], 3, axis=2)
    class_rgb[...,2] = 0
    class_rgb[...,1] = target

    input_im = np.hstack(input_image.cpu().numpy().transpose(0,2,3,1))
    
    if input_im.shape[2] == 3:
        input_im_grayscale = np.repeat(input_im.mean(axis=2)[..., None], 3, axis=2)
        overlayed_im = (input_im_grayscale*0.6 + class_rgb*0.7).clip(0,1)
    else:
        input_map = input_im[...,3:]
        overlayed_im = (input_map*0.6 + class_rgb*0.7).clip(0,1)

    thresholded_pred = np.repeat(class_one_preds[..., None] > 0.5, 3, axis=2)

    fig = plt.figure(figsize=(12,26))
    plot_im = np.vstack([class_rgb, input_im[...,:3], overlayed_im, thresholded_pred]).clip(0,1).astype(np.float32)
    plt.imshow(plot_im)
    plt.axis("off")
    plt.savefig(os.path.join(predictions_dir,"preds_{}_epoch_{}.png".format(model_name, epoch)))


def main():
                
    set_seed(seed=1234)

    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]

    model = get_unet_model(num_output_classes=len(classes)+1) # one more class for background, 0 on the masks
    data_folder = os.path.join(OUTPUT_ROOT, "bev_data")

    checkpoint= f'{OUTPUT_ROOT}/checkpoints/unet_4_64_320_fold_3/unet_4_64_320_fold_3_epoch_4.pth'

    train(model, model_name='unet_4_64_384', data_folder=data_folder, 
          fold=3, debug=False, img_size=IMG_SIZE,
          epochs=25, batch_size=16, num_workers=4, resume_weights=checkpoint, resume_epoch=4)


if __name__ == "__main__":
    main()

    
