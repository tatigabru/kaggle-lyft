# -*- coding: utf-8 -*-
"""
An Instance segmentation model for Lyft Dataset
In our case, we want to fine-tune from a pre-trained on coco 
Mask-RCNN model on our dataset. 

"""
import argparse
import collections
import datetime
import glob
import os
import pickle
import random
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from skimage.color import label2rgb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
# my imports
from coco_helpers.my_engine import evaluate, train_one_epoch
from coco_helpers.utils import collate_fn
from configs import (BEV_SHAPE, DATA_ROOT, IMG_SIZE, NUM_CLASSES, ON_SERVER,
                     OUTPUT_ROOT, PROJECT_ROOT)
from datasets.bev_dataset_coco import BevDatasetAugs
from datasets.transforms import (D4_transforms, augment_and_show,
                                 train_transforms, valid_transforms,
                                 visualize_bbox)
# lyft SDK imports
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import (Box, LidarPointCloud,
                                                 Quaternion)
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix, view_points
from models.models import get_maskrcnn_model
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from utilities.utils import set_seed

NUM_CLASSES = NUM_CLASSES + 1 # + 1 for background
SAVE_PATH = OUTPUT_ROOT + '/maskrcnn'


def load_model_optim(model, optimizer, checkpoint_path: str):
    """Loads model weigths, optimizer, epoch to continuer training
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    print('Loaded model from {}, epoch {}'.format(checkpoint_path, epoch))


def train(model, model_name: str, data_folder: str, level5data, 
          fold: int, debug=False, img_size=IMG_SIZE, bev_shape=BEV_SHAPE,
          epochs=15, batch_size = 8, num_workers=4, learning_rate=1e-3, resume_weights='', resume_epoch=0):
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
    print(f'device: {device}')

    # weight loss for the 0 class lower to account for (some of) the big class imbalance
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

    # BEV conversion parameters    
    voxel_size = (0.2, 0.2, 1.5)
    z_offset = -2.0    

    # choose inputs/targets
    input_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_input.png")))
    sample_tokens = [x.split("/")[-1].replace("_input.png","") for x in input_filepaths]
    sample_tokens = [x.replace("bev_data\\","") for x in sample_tokens]    

    # train samples
    df = pd.read_csv('folds/train_samples.csv')
    train_df = df[df['samples'].isin(sample_tokens)]
    print('train samples: ', train_df.head())

    # validation samples
    df = pd.read_csv('folds/val_samples.csv')
    valid_df = df[df['samples'].isin(sample_tokens)]
    print('valid samples: ', valid_df.head())
    
    # load weights to continue training
    if resume_weights != '':
        print('Load model from: {}'.format(resume_weights))
        checkpoint = torch.load(resume_weights)
        model.load_state_dict(checkpoint['model'])
        resume_epoch = checkpoint['epoch']+1      	
    
    model.to(device)

    # optimizer and schedulers    
    print(f'learning_rate: {learning_rate}')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)  

    # datasets for train and validation 
    train_dataset = BevDatasetAugs(fold=fold, df=train_df, 
                                    level5data = level5data,
                                    debug=debug, 
                                    img_size=bev_shape[0], 
                                    input_dir=data_folder, 
                                    transforms = train_transforms,                                    
                                    bev_shape = bev_shape,
                                    voxel_size = voxel_size, 
                                    z_offset = z_offset)

    valid_dataset = BevDatasetAugs(fold=fold, df=valid_df, 
                                    level5data = level5data,
                                    debug=debug, 
                                    img_size=bev_shape[0], 
                                    input_dir=data_folder, 
                                   # transforms = valid_transforms,                                    
                                    bev_shape = bev_shape,
                                    voxel_size = voxel_size, 
                                    z_offset = z_offset)

    # dataloaders for train and validation
    train_loader = DataLoader(train_dataset, 
                              num_workers=num_workers,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)

    valid_loader = DataLoader(valid_dataset,
                              num_workers=num_workers,
                              batch_size=1,
                              shuffle=False,
                              collate_fn=collate_fn)

    print('{} training images, {} validation images'.format(len(train_dataset), len(valid_dataset)))

    # training cycle           
    print("Start training")
    start_time = time.time()
    for epoch in range(epochs):                
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, None, train_loader, device, epoch, print_freq=10, warmup = True)
        #lr_scheduler.step()

        # save model after every epoch
        torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        #'lr_scheduler': scheduler.state_dict(),
        'epoch': epoch,        
        }, os.path.join(checkpoints_dir, f'{model_name}_fold_{fold}_epoch_{epoch}.pth'))         
        	       
        # validate model after every epoch
        #evaluate(model, valid_loader, device=device)
        #valid_loss = validate(model, valid_loader, class_weights,
        #                      epoch, predictions_dir, save_oof = True)
        
   
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    return model


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

        for iter_num, (img, target) in enumerate(progress_bar):
            img = img.to(device)  # [N, 3, H, W]
            target = target.to(device)  # [N, H, W] with class indices (0, 1)
            prediction = model(img)  # [N, 2, H, W]
            loss = F.cross_entropy(prediction, target, weight=class_weights)
            val_losses.append(loss.detach().cpu().numpy())                  
    print("Epoch {}, Valid Loss: {}".format(epoch, np.mean(val_losses)))
        
    return np.mean(val_losses)   


def predict(model, dataset_test):    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img, _ = dataset_test[0]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])         
    return prediction  


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

    # get data
    if ON_SERVER:
        level5data = LyftDataset(data_path = '.', json_path='../../input/train_data', verbose=True) # server
    else:
        level5data = LyftDataset(data_path = '../input/', json_path='../input/train_data', verbose=True) # local laptop
    
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    
    # "bev" folder
    data_folder = os.path.join(OUTPUT_ROOT, "bev_data")

    # choose model
    model = get_maskrcnn_model(NUM_CLASSES)     
    
    checkpoint= f'{OUTPUT_ROOT}/checkpoints/'

    train(model, model_name='mask_512', data_folder=data_folder, 
          level5data = level5data, 
          fold=args.fold, debug=args.debug, img_size=IMG_SIZE, bev_shape=BEV_SHAPE,
          epochs=args.epochs, batch_size=16, num_workers=4, 
          learning_rate = args.lr, resume_weights=args.weights_dir, resume_epoch=0)

      
if __name__ == '__main__':   
    main()  
