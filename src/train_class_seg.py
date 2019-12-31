import argparse
import collections
import glob
import os
import pickle
import random
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import scipy.special
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm, tqdm_notebook

# extra libruaries
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from configs import IMG_SIZE, NUM_CLASSES, OUTPUT_ROOT
# current project imports 
from datasets.biv_dataset import BEVImageDataset, BEVLabelsDataset
from datasets.transforms import (albu_valid_tansforms, crop_d4_transforms,
                                 tensor_transform, test_transform,
                                 train_transform)
# lyft SDK imports
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import (Box, LidarPointCloud,
                                                 Quaternion)
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix, view_points
from models.models import (
    get_fpn_model, get_fpn_twohead_model, get_smp_model, get_unet_model,
    get_unet_twohead_model)
from pytorch_toolbelt import losses as L
from torch import optim
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from utilities.iou import iou_numpy
from utilities.logger import Logger
from utilities.radam import RAdam
from utilities.utils import set_seed

os.environ["OMP_NUM_THREADS"] = "1"


def load_model_optim(model, optimizer, checkpoint_path: str):
    """Loads model weigths, optimizer, epoch, step abd loss to continuer training
    """  
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']+1
    model.cuda()
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()    
    for param_group in optimizer.param_groups:
        print('learning_rate:', param_group['lr'])
    print('Loaded model from {}, epoch {}'.format(checkpoint_path, epoch))


def load_model(model, checkpoint_path: str):
    """Loads model weigths, optimizer, epoch, step abd loss to continuer training
    """  
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']+1
    print('Loaded model from {}, epoch {}'.format(checkpoint_path, epoch))
 
 
def train(model, model_name: str, data_folder: str, fold: int, debug=False, img_size=IMG_SIZE,
          learning_rate=1e-3, epochs=15, batch_size = 8, num_workers=4, resume_weights='', resume_epoch=0):
    """
    Model training
    
    Args: 
        model : PyTorch model
        model_name : string name for model for checkpoints saving
        fold: evaluation fold number, 0-3
        debug: if True, runs the debugging on few images 
        img_size: size of images for training (for pregressive learning)
        learning_rate: initial learning rate
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
    tensorboard_dir = f'{OUTPUT_ROOT}/tensorboard/{model_name}_fold_{fold}'
    validations_dir = f'{OUTPUT_ROOT}/oof/{model_name}_fold_{fold}/val'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(validations_dir, exist_ok=True)
    logger = Logger(tensorboard_dir)

    print('\n', model_name, '\n')

    # choose inputs/targets
    input_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_input.png")))
    sample_tokens = [x.split("/")[-1].replace("_input.png","") for x in input_filepaths]
    sample_tokens = [x.replace("bev_data\\","") for x in sample_tokens]    

    # train samples
    df = pd.read_csv(f'folds/train_fold_{fold}.csv')
    train_df = df[df['samples'].isin(sample_tokens)]
    print('train samples: ', train_df.head())

    # validation samples
    df = pd.read_csv(f'folds/val_fold_{fold}.csv')
    valid_df = df[df['samples'].isin(sample_tokens)]
    print('valid samples: ', valid_df.head())
    
    # optimizer and schedulers
    print(f'initial learning rate: {learning_rate}')
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = RAdam(model.parameters(), lr=learning_rate)
    for param_group in optimizer.param_groups:
        print('learning_rate:', param_group['lr'])
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.2)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6], gamma=0.2)
    
    # load model weights to continue training
    if resume_weights != '':
        load_model(model, resume_weights)                  	
    model = model.to(device) 
   
  
    # datasets for train and validation
    train_dataset = BEVLabelsDataset(fold=fold, df=train_df, 
                                    debug=debug, img_size=img_size, 
                                    input_dir=data_folder,
                                    transforms = crop_d4_transforms)
    
    valid_dataset = BEVLabelsDataset(fold=fold, df=valid_df, 
                                    debug=debug, img_size=img_size, 
                                    input_dir=data_folder,
                                    transforms = albu_valid_tansforms)                                

    # dataloaders for train and validation
    dataloader_train = DataLoader(train_dataset, 
                                  num_workers=num_workers,
                                  batch_size=batch_size,
                                  shuffle=True)
    
    dataloader_valid = DataLoader(valid_dataset,
                                  num_workers=num_workers,
                                  batch_size=8,
                                  shuffle=False)

    print('{} training images, {} validation images'.format(len(train_dataset), len(valid_dataset)))
    
    # training cycle
    print("Start training")
    train_loss_cls, train_loss_seg, valid_loss_cls, valid_loss_seg = [], [], [], []
    history = {}
        
    for epoch in range(resume_epoch, epochs+1):
        print("Epoch", epoch)        
        cls_losses, seg_losses = [], []
        progress_bar = tqdm(dataloader_train, total=len(dataloader_train))

        #with torch.set_grad_enabled(True): --> sometime people write it
        for iter_num, (img, target, labels, sample_ids) in enumerate(progress_bar): im, target,  sample_token
            img = img.to(device)  # [N, 3, H, W]
            target = target.to(device)  # [N, H, W] with class indices (0, 1)
            (mask_pred, cls_pred) = model(img)  # [N, 2, H, W] | prediction
            loss_seg = F.cross_entropy(mask_pred, target, weight=class_weights) 
            loss_cls = F.cross_entropy(cls_pred, labels, weight=class_weights)     
            loss = loss_cls + loss_seg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 
            optimizer.step()   
                        
            cls_losses.append(loss_cls.detach().cpu().numpy())
            seg_losses.append(loss_seg.detach().cpu().numpy())
            
            if iter_num == 0:
                visualize_predictions(img, mask_pred, target, predictions_dir, model_name, epoch)

        # loss history
        print(f"Epoch {epoch}, Train Classification Loss: {np.mean(cls_losses)}, Train Segmentation Loss: {np.mean(seg_losses)}")
        train_loss_cls.append(np.mean(cls_losses)) 
        train_loss_seg.append(np.mean(seg_losses))
        logger.scalar_summary('loss_train', np.mean(cls_losses), epoch)
        logger.scalar_summary('loss_train', np.mean(seg_losses), epoch)
                
        # validate model afterevery epoch
        valid_loss = validate(model, model_name, dataloader_valid, class_weights,
                              epoch, validations_dir, save_oof = True)
        valid_losses.append(valid_loss)
        logger.scalar_summary('loss_valid', valid_loss, epoch)  
        
        # print current learning rate
        for param_group in optimizer.param_groups:
            print('learning_rate:', param_group['lr'])
        scheduler.step()
        
        # save model, optimizer and scheduler after every epoch
        checkpoint_filename = "{}_fold_{}_epoch_{}.pth".format(model_name, fold, epoch)
        checkpoint_filepath = os.path.join(checkpoints_dir, checkpoint_filename)
        torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': np.mean(epoch_losses),
        'valid_loss': valid_loss,
         }, checkpoint_filepath)  
         


def validate(model, model_name: str, dataloader_valid, class_weights, epoch: int, 
             validations_dir: str, save_oof=True):
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
        val_cls_losses, val_seg_losses = [], []        
        progress_bar = tqdm(dataloader_valid, total=len(dataloader_valid))

        for iter_num, (img, target, sample_ids) in enumerate(progress_bar):
            img = img.to(device)  # [N, 3, H, W]
            target = target.to(device)  # [N, H, W] with class indices (0, 1)
            (mask_pred, cls_pred) = model(img)  # [N, 2, H, W] | prediction
            loss_seg = F.cross_entropy(mask_pred, target, weight=class_weights) 
            loss_cls = F.cross_entropy(cls_pred, labels, weight=class_weights)     
            val_seg_losses.append(loss_seg.detach(),cpu(),numpy())
            val_cls_losses.append(loss_cls.detach().cpu().numpy())
      
            # Visualize the first prediction
            if iter_num == 0:
                visualize_predictions(img, mask_pred, target, validations_dir, model_name, epoch)   
                                
    print(f"Epoch {epoch}, Valid Classification Loss: {np.mean(val_cls_losses)}, 
            Train Segmentation Loss: {np.mean(val_seg_losses)}")
            
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
        predictions: the predictions thresholded at 0.5 probability
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
    parser = argparse.ArgumentParser()                                
    arg = parser.add_argument    
    arg('--model_name', type=str, default='mask_512', help='String model name from models dictionary')
    arg('--seed', type=int, default=1234, help='Random seed')
    arg('--fold', type=int, default=0, help='Validation fold')
    arg('--weights_dir', type=str, default='', help='Directory for loading model weights')
    arg('--epochs', type=int, default=12, help='Current epoch')
    arg('--lr', type=float, default=1e-3, help='Initial learning rate')
    arg('--debug', type=bool, default=False, help='If the debugging mode')
    args = parser.parse_args() 

    set_seed(args.seed)
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    model = get_unet_twohead_model(encoder='resnet18', num_classes=len(classes)+1)
    
    data_folder = os.path.join(OUTPUT_ROOT, "bev_data")

    checkpoint= f'{OUTPUT_ROOT}/checkpoints/unet_resnet152_384_fold_2/unet_resnet152_384_fold_2_epoch_6.pth' 
    
    train(model, model_name=args.model_name, data_folder=data_folder, 
          fold=args.fold, debug=args.debug, img_size=IMG_SIZE, learning_rate= args.lr,
          epochs=args.epochs, batch_size=32, num_workers=4, resume_weights='', resume_epoch=0)



if __name__ == "__main__":
    main()
