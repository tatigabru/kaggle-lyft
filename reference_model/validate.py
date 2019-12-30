"""

Validation

"""
import argparse
import collections
import pickle
import random
import pandas as pd
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
from configs import OUTPUT_ROOT, IMG_SIZE
from transforms import (train_transform, test_transform, tensor_transform,
                        crop_d4_transforms, albu_valid_tansforms)
from metric import map_sample
from train import get_unet_model, set_seed, visualize_predictions



def get_unet_model(in_channels=3, num_output_classes=2):
    model = UNet(in_channels=in_channels, n_classes=num_output_classes, wf=5, depth=4, padding=True, up_mode='upsample')    
    # Optional, for multi GPU training and inference
    #model = nn.DataParallel(model)
    return model


def load_model(model, checkpoint_path: str):
    """Loads model weigths, epoch, and val_loss
    """   
    #checkpoint = torch.load(checkpoint_path)     
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  
    model.load_state_dict(checkpoint['model'])    
    epoch = checkpoint['epoch']
    val_loss = checkpoint['valid_loss']
    print('Loaded model from {}, epoch {}. Validation loss: {}'.format(checkpoint_path, epoch, val_loss))
    
    return model


def generate_predictions(model: nn.Module, checkpoint_path: str, dataloader_valid, 
                        predictions_dir: str, img_size = IMG_SIZE, save_oof=True):
    """
    Loads model checkpoints, 
    calculates oof predictions for and saves them to pickle
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    load_model(model, checkpoint_path)
    os.makedirs(predictions_dir, exist_ok=True)
    
    progress_bar = tqdm(dataloader_valid, total=len(dataloader_valid))
    # We quantize to uint8 here to conserve memory. We're allocating >20GB of memory otherwise.
    predictions = np.zeros((len(valid_dataset), 1+len(classes), img_size, img_size), dtype=np.uint8)
    targets = np.zeros((len(target_filepaths), img_size, img_size), dtype=np.uint8)

    sample_tokens = []
    with torch.no_grad():
        model.eval()        
        for iter_num, (img, target, sample_ids) in enumerate(progress_bar):
            offset = iter_num*batch_size            
            sample_tokens.extend(sample_ids)

            img = img.to(device)  # [N, 3, H, W]
            prediction = model(img)  # [N, 2, H, W] 
            oof = prediction.cpu().numpy()
            prediction = F.softmax(prediction, dim=1)
            # model outputs to numpy
            prediction_cpu = prediction.cpu().numpy()
            predictions[offset:offset+batch_size] = np.round(prediction_cpu*255).astype(np.uint8)
            targets[offset:offset+batch_size] = target.numpy()

            # save logits for folds
            if save_oof:                   
                np.save(f'{predictions_dir}/{sample_ids}_oof_epoch_{epoch:03}.npy', oof)
             
            # Visualize the first prediction
            if iter_num == 0:
                visualize_predictions(img, prediction, target, apply_softmax=False)              
            
        if save_oof:
            np.save(f'{predictions_dir}/predictions_epoch_{epoch:03}.npy', predictions)

    return targets, predictions 


def validate_metric(model: nn.Module, checkpoint_path: str, dataloader_valid, predictions_dir: str, save_oof=True):
    """
    Calculate validation metric in numpy

    Input: 
        model: current model 
        checkpoint_path: model checkpoint filepath
        dataloader_valid: dataloader for the validation fold
        predictions_dir: directory for saving predictions
        save_oof: boolean flag, if calculate oof predictions and save them in pickle 
               
    Output:
        valid_metric: mAP cumulated for samplesand averaged
    """ 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    load_model(model, checkpoint_path)

    with torch.no_grad():
        model.eval()
        val_metrics = []
        progress_bar = tqdm(dataloader_valid, total=len(dataloader_valid))

        for iter_num, (img, target, sample_ids) in enumerate(progress_bar):
            img = img.to(device)  # [N, 3, H, W]
            target = target.to(device)  # [N, H, W] with class indices (0, 1)
            output = model(img)  # [N, 2, H, W]
            prediction = F.softmax(output, dim=1)
            # model outputs to numpy
            prediction = prediction.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            val_metrics.append(map_sample(target, prediction))

        print("Epoch {}, mean Average Precision: {}".format(epoch, np.mean(val_metrics)))
        
    return np.mean(val_metrics)    


def optimise_thresholds(thresholds):
    for thres in thresholds:
        predicted_mask = outputs > thres  # This will be a byte tensor
        iou_pytorch(predicted_mask, labels.byte())  # If your labels are BATCH x H x W
        # iou_pytorch(predicted_mask, labels.squeeze(1).byte())  


def check_thresholds(predictions, background_threshold = 255//2):
    """# Arbitrary threshold in our system to create a binary image to fit boxes around"""
    # Get probabilities for non-background
    predictions_non_class0 = 255 - predictions[:,0]

    for i in range(3):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
        axes[0].imshow(predictions_non_class0[i])
        axes[0].set_title("predictions")
        axes[1].imshow(predictions_non_class0[i] > background_threshold)
        axes[1].set_title("thresholded predictions")
        axes[2].imshow((targets[i] > 0).astype(np.uint8), interpolation="nearest")
        axes[2].set_title("targets")
        fig.tight_layout()
        fig.show()

def seaparate_instances(predictions_non_class0, background_threshold = 255//2):
    # We perform an opening morphological operation to filter tiny detections
    # Note that this may be problematic for classes that are inherently small (e.g. pedestrians)..
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    predictions_opened = np.zeros((predictions_non_class0.shape), dtype=np.uint8)

    for i, p in enumerate(tqdm(predictions_non_class0)):
        thresholded_p = (p > background_threshold).astype(np.uint8)
        predictions_opened[i] = cv2.morphologyEx(thresholded_p, cv2.MORPH_OPEN, kernel)

    plt.figure(figsize=(12,12))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    axes[0].imshow(predictions_non_class0[0] > 255//2)
    axes[0].set_title("thresholded prediction")
    axes[1].imshow(predictions_opened[0])
    axes[1].set_title("opened thresholded prediction")
    fig.show()

def main():

    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]

    model = get_unet_model(num_output_classes=len(classes)+1) # one more class for background, 0 on the masks

    data_folder = os.path.join(OUTPUT_ROOT, "bev_data")
    checkpoint= f'{OUTPUT_ROOT}/checkpoints/unet_768_fold_3/unet_768_fold_3_epoch_5.pth'   
    predictions_dir = predictions_dir = f'{OUTPUT_ROOT}/oof/unet_768_fold_3'
    load_model(model, checkpoint) 

    # choose inputs/targets
    input_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_input.png")))
    sample_tokens = [x.split("/")[-1].replace("_input.png","") for x in input_filepaths]
    sample_tokens = [x.replace("bev_data\\","") for x in sample_tokens]    

    # validation samples
    df = pd.read_csv('val_samples.csv')
    valid_df = df[df['samples'].isin(sample_tokens)]
    print('valid samples: ', valid_df.head())

    valid_dataset = BEVImageDataset(fold=3, df=valid_df, 
                                    debug=True, img_size=IMG_SIZE, 
                                    input_dir=data_folder,
                                    transforms = None)    

    dataloader_valid = DataLoader(valid_dataset,
                                  num_workers=2,
                                  batch_size=4,
                                  shuffle=False)   

    val_metric = validate_metric(model, checkpoint, dataloader_valid, predictions_dir, save_oof=False)
    print(val_metric)
    print("Mission accomplished")       


if __name__ == "__main__":
    main()                                                         