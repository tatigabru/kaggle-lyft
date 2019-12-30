import argparse
import collections
import pickle
import random
import pandas as pd
from datetime import datetime
from functools import partial
import glob
import logging
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
from pytorch_toolbelt import losses as L
os.environ["OMP_NUM_THREADS"] = "1"

# lyft SDK imports
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

# extra libruaries
import segmentation_models_pytorch as smp
from pytorch_toolbelt import losses as L

# current project imports 
from utilities.logger import Logger
from unet import UNet
from datasets.BEVDataset import BEVImageDataset
from configs import OUTPUT_ROOT, IMG_SIZE, NUM_CLASSES
from transforms import (train_transform, test_transform, tensor_transform,
                        crop_d4_transforms, albu_valid_tansforms)

from iou import iou_numpy

from train import set_seed, get_unet_model


def visualize_validation(model, dataloader_valid, class_weights, epoch: int, 
             predictions_dir: str, save_oof=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)    
    with torch.no_grad():
        model.eval()
        val_losses = []
        val_metric = []
        progress_bar = tqdm(dataloader_valid, total=len(dataloader_valid))   

        for iter_num, (img, target, sample_ids) in enumerate(progress_bar):
            img = img.to(device)  # [N, 3, H, W]
            target = target.to(device)  # [N, H, W] with class indices (0, 1)
            prediction = model.forward(img)  
            prediction = F.softmax(prediction, dim=1)
            # Visualize some predictions
            if iter_num % 550 == 0:
                visualize_predictions(img, prediction, target, apply_softmax=False)            
    print("Validation predictions plotted")


def validate_jaccard(model: nn.Module, dataloader_valid, class_weights):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Validation on hold-out....")
    with torch.no_grad():
        model.eval()
        val_losses = []
        val_metric, jaccard = []
        progress_bar = tqdm(dataloader_valid, total=len(dataloader_valid))

        for iter_num, (inputs, targets, sample_ids) in enumerate(progress_bar):  
            inputs = variable(inputs, volatile=True)
            targets = variable(targets)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, weight=class_weights)
            val_losses.append(loss.detach().cpu().numpy())
            jaccard += [get_jaccard(targets, (outputs > 0).float()).data[0]]

    valid_loss = np.mean(val_losses)  # type: float
    valid_jaccard = np.mean(jaccard)
    print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
    metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard}
    
    return metrics

# helper functions
def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))

def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x

def get_jaccard(y_true, y_pred):
    y_pred = torch.nn.Softmax2d()
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1).sum(dim = -1)
    union = y_true.sum(dim=-2).sum(dim=-1).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1).sum(dim = -1)
    return (intersection / (union - intersection + epsilon)).mean()


def iou_numpy(outputs: np.array, labels: np.array, num_classes = 9):
    """My multicalss iou from 
       Numpy version
    """
    SMOOTH = 1e-6
    ious = []
    #outputs = outputs.squeeze(1)
    for num in range(1, num_classes+1):
        intersection = ((outputs==num) * (labels==num)).sum()
        union = (outputs==num).sum() + (labels==num).sum() - intersection
        if union == 0: 
            ious.append(float('nan'))  # if there is no class in ground truth, do not include in evaluation
        else:  
            ious.append((intersection + SMOOTH) / (union + SMOOTH))    
    #thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return np.array(ious), np.nanmean(ious) 

