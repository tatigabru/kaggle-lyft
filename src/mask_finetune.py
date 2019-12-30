# -*- coding: utf-8 -*-
"""
An Instance segmentation model for Lyft Dataset
In our case, we want to fine-tune from a pre-trained coco model on our dataset. 

We will be using Mask R-CNN

"""
import collections
import datetime
import glob
import os
import time
import argparse
import pickle
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision

# lyft SDK imports
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import (Box, LidarPointCloud,
                                                 Quaternion)
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix, view_points
from PIL import Image, ImageFile
from skimage.color import label2rgb
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm
import sys

sys.path.append('C:/Users/New/Documents/Challenges/lyft/')
sys.path.append('C:/Users/New/Documents/Challenges/lyft/progs/')

# my imports
from progs.datasets.bev_dataset_augs import BevDatasetAugs
from progs.coco_helpers.utils import collate_fn
from progs.coco_helpers.my_engine import evaluate, train_one_epoch
from progs.configs import BEV_SHAPE, DATA_ROOT, IMG_SIZE, NUM_CLASSES, OUTPUT_ROOT
from progs.transforms import (D4_transforms, albu_valid_tansforms, augment_and_show,
                        train_transforms, visualize_bbox)
from progs.utils.my_utils import set_seed
from progs.models import get_maskrcnn_model
le = LabelEncoder()

set_seed(seed=1234)

NUM_CLASSES = NUM_CLASSES + 1 # + 1 for background
SAVE_PATH = OUTPUT_ROOT + '/maskrcnn'
N_EPOCHS = 10


def load_model_optim(model, optimizer, checkpoint_path: str):
    """Loads model weigths, optimizer, epoch, step abd loss to continuer training
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    print('Loaded model from {}, epoch {}'.format(checkpoint_path, epoch))


def train(model, model_name: str, data_folder: str, fold: int, debug=False, img_size=IMG_SIZE,
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
    print(device)

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

    # choose inputs/targets
    input_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_input.png")))
    sample_tokens = [x.split("/")[-1].replace("_input.png","") for x in input_filepaths]
    sample_tokens = [x.replace("bev_data\\","") for x in sample_tokens]    

    # train samples
    df = pd.read_csv('../folds/train_samples.csv')
    train_df = df[df['samples'].isin(sample_tokens)]
    print('train samples: ', train_df.head())

    # validation samples
    df = pd.read_csv('../folds/val_samples.csv')
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
    print(learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    #lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.2)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)  

    # datasets for train and validation 
    train_dataset = BevDatasetAugs(fold=0, df=train_df, 
                                    level5data = level5data,
                                    debug=True, 
                                    img_size=bev_shape[0], 
                                    input_dir=data_folder, 
                                    transforms = train_transforms,                                    
                                    bev_shape = bev_shape,
                                    voxel_size = voxel_size, 
                                    z_offset = z_offset)

    valid_dataset = BevDatasetAugs(fold=0, df=train_df, 
                                    level5data = level5data,
                                    debug=True, 
                                    img_size=bev_shape[0], 
                                    input_dir=data_folder, 
                                    transforms = albu_valid_tansforms,                                    
                                    bev_shape = bev_shape,
                                    voxel_size = voxel_size, 
                                    z_offset = z_offset)

    # dataloaders for train and validation
    train_loader = DataLoader(train_dataset, 
                              num_workers=num_workers,
                              batch_size=batch_size,
                              shuffle=True)

    valid_loader = DataLoader(valid_dataset,
                              num_workers=num_workers,
                              batch_size=4,
                              shuffle=False)
    print('{} training images, {} validation images'.format(len(train_dataset), len(valid_dataset)))

    # training cycle           
    print("Start training")
    start_time = time.time()
    for epoch in range(num_epochs):                
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, lr_scheduler, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        # save model after every epoch
        torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,        
        }, os.path.join(model_path, 'model_{}.pth'.format(epoch)))         
        evaluate(model, valid_loader, device=device)
   
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    return model

def main():
    # get data
    level5data = LyftDataset(data_path = '../input/', json_path='../input/train_data', verbose=True) # local laptop
    #level5data = LyftDataset(data_path = '.', json_path='../../input/train_data', verbose=True) # server
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    
    # BEV conversion parameters
    bev_shape = (768, 768, 3)
    voxel_size = (0.2, 0.2, 1.5)
    z_offset = -2.0
    box_scale = 0.8
    img_size = IMG_SIZE

    # "bev" folders
    data_folder = os.path.join(OUTPUT_ROOT, "bev_data")

    # choose model
    model = get_maskrcnn_model(NUM_CLASSES)     




def predict(model, dataset_test):    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img, _ = dataset_test[0]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])         
    return prediction    


model = train(data_loader, data_loader_test, model_path = SAVE_PATH, learning_rate = 1e-4, num_epochs = N_EPOCHS)      
