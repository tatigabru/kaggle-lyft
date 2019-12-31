"""
From Lyft Reference_model

"""
import glob
import os
import sys
sys.path.append('/home/user/challenges/lyft/lyft_repo/src')

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from configs import DATA_ROOT, IMG_SIZE, NUM_CLASSES, OUTPUT_ROOT
from datasets.transforms import (albu_show_transforms, albu_valid_tansforms,
                                 crop_d4_transforms)
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import (Box, LidarPointCloud,
                                                 Quaternion)
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix, view_points


class BEVImageDataset(torch.utils.data.Dataset):
    """
    Bird-eye-view dataset

    Args:
        fold: integer, number of the fold
        df: Dataframe with sample tokens
        debug: if True, runs the debugging on few images
        img_size: the desired image size to resize to        
        input_dir: directory with imputs and targets (and maps, optionally)
        transforms: augmentations (albumentations composed list))  
        """        
    def __init__(self, fold: int, df: pd.DataFrame, 
                 debug: bool, img_size: int, 
                 input_dir: str, transforms = None):
        super(BEVImageDataset, self).__init__()  # inherit it from torch Dataset
        self.fold = fold
        self.df = df
        self.debug = debug
        self.img_size = img_size
        self.input_dir = input_dir
        self.transforms = transforms
        
        if self.debug:
            self.df = self.df.head(16)
            print('Debug mode, samples: ', self.df.samples)  

        self.sample_tokens = list(self.df.samples)
        
    def __len__(self):
        return len(self.sample_tokens)
    
    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]
        input_filepath = '{}/{}_input.png'.format(self.input_dir, sample_token)
        target_filepath = '{}/{}_target.png'.format(self.input_dir, sample_token)
        
        im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)  
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)      
        target = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)

        if self.transforms is not None: 
            augmented = self.transforms(image=im, mask=target)  
            im = augmented['image']
            target = augmented['mask']
        else: 
            im = cv2.resize(im, (self.img_size, self.img_size))
            target = cv2.resize(target, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            im = im.astype(np.float32)/255
        
        target = target.astype(np.int64)    
        
        im = torch.from_numpy(im.transpose(2,0,1)) # channels first
        target = torch.from_numpy(target)  # single channel 

        return im, target, sample_token


class BEVLabelsDataset(torch.utils.data.Dataset):
    """
    Bird-eye-view dataset with labels 
    for two headed models

    Args:
        df: Dataframe with sample tokens
        debug: if True, runs the debugging on few images
        img_size: the desired image size to resize to        
        input_dir: directory with imputs and targets (and maps, optionally)
        num_classes: numbr of classes in classification head
        transforms: augmentations (albumentations composed list))        
    """        
    def __init__(self, df: pd.DataFrame, 
                 debug: bool, img_size: int, 
                 input_dir: str, 
                 num_classes = NUM_CLASSES,
                 transforms = None):
        super(BEVLabelsDataset, self).__init__()  # inherit it from torch Dataset
        self.df = df
        self.debug = debug
        self.img_size = img_size
        self.input_dir = input_dir
        self.num_classes = num_classes
        self.transforms = transforms
        
        if self.debug:
            self.df = self.df.head(16)
            print('Debug mode, samples: ', self.df.samples)  

        self.sample_tokens = list(self.df.samples)
        
    def __len__(self):
        return len(self.sample_tokens)
    
    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]
        input_filepath = '{}/{}_input.png'.format(self.input_dir, sample_token)
        target_filepath = '{}/{}_target.png'.format(self.input_dir, sample_token)
        
        im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)  
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)      
        target = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)
     
        if self.transforms is not None: 
            augmented = self.transforms(image=im, mask=target)  
            im = augmented['image']
            target = augmented['mask']
        else: 
            im = cv2.resize(im, (self.img_size, self.img_size))
            target = cv2.resize(target, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            im = im.astype(np.float32)/255

        target = target.astype(np.int64)      

        # get classes
        img_classes = np.unique(target)
        print(f'img_classes {img_classes}') # sorted unique elements 
        # make labels one-hot encoding, obey class 0
        labels = torch.zeros(self.num_classes)
        for cls in img_classes[1:]: 
            labels[int(cls)] = int(1)    
        
        im = torch.from_numpy(im.transpose(2,0,1)) # channels first
        target = torch.from_numpy(target)  # single channel 

        return im, target, labels, sample_token


class BEVMapsDataset(torch.utils.data.Dataset):
    """
    Bird-eye-view dataset with maps

    Args:
        df: Dataframe with sample tokens
        debug: if True, runs the debugging on few images
        img_size: the desired image size to resize to        
        input_dir: directory with inputs and targets (and maps, optionally)
        maps_dir: directory with input maps
        num_classes: numbr of classes in classification head
        transforms: augmentations (albumentations composed list))        
    """        
    def __init__(self, df: pd.DataFrame, 
                 debug: bool, img_size: int, 
                 input_dir: str, maps_dir: str, 
                 num_classes = NUM_CLASSES, transforms = None):
        super(BEVMapsDataset, self).__init__()  # inherit it from torch Dataset    
        self.df = df
        self.debug = debug
        self.img_size = img_size
        self.input_dir = input_dir
        self.maps_dir = maps_dir
        self.num_classes = num_classes
        self.transforms = transforms
        
        if self.debug:
            self.df = self.df.head(16)
            print('Debug mode, samples: ', self.df.samples)  
        self.sample_tokens = list(self.df.samples)
        
    def __len__(self):
        return len(self.sample_tokens)
    
    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]
        input_filepath = '{}/{}_input.png'.format(self.input_dir, sample_token)
        target_filepath = '{}/{}_target.png'.format(self.input_dir, sample_token)
        map_filepath = '{}/{}_map.png'.format(self.maps_dir, sample_token) 

        im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)  
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)      
        target = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)
        map_im = cv2.imread(map_filepath, cv2.IMREAD_UNCHANGED)
        map_im = cv2.resize(map_im, (768, 768))         
        
        im  = 2*im//3 + map_im//3 # as alternative, we can just add map overlay on image  
        if self.transforms is not None: 
            augmented = self.transforms(image=im, mask=target)  
            im = augmented['image']
            target = augmented['mask']
        
        im = cv2.resize(im, (self.img_size, self.img_size))
        target = cv2.resize(target, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        # print(im.shape, map_im.shape)
        #im = np.concatenate((im, map_im), axis=2)  # second option is concatenate
        #im = im.astype(np.float32)/255
        
        target = target.astype(np.int64) 
        # get classes
        img_classes = np.unique(target)
        print(f'img_classes {img_classes}')
        # make labels one-hot encoding
        labels = torch.zeros(num_classes)
        for cls in img_classes[1:]:
            labels[int(cls)] = int(1)    

        im = torch.from_numpy(im.transpose(2,0,1)) # channels first
        target = torch.from_numpy(target)  # single channel 

        return im, target, labels, sample_token

"""

TEST DATASETS

"""

class BEVTestDataset(torch.utils.data.Dataset):
    """
    Bird-eye-view dataset, test

    Args:         
        sample_tokens: list with sample tokens
        debug: if True, runs the debugging on few images
        img_size: the desired image size to resize to        
        input_dir: directory with imputs and targets (and maps, optionally)        
        transforms: augmentations (albumentations composed list))        
    """        
    def __init__(self, sample_tokens: list, 
                 debug: bool, img_size: int, 
                 input_dir: str, transforms = None):
        super(BEVTestDataset, self).__init__()  # inherit it from torch Dataset        
        self.sample_tokens = sample_tokens
        self.debug = debug
        self.img_size = img_size
        self.input_dir = input_dir        
        self.transforms = transforms
        
        if self.debug:
            self.sample_tokens = self.sample_tokens[:16]
            print('Debug mode, samples: ', self.sample_tokens)
                
    def __len__(self):
        return len(self.sample_tokens)
    
    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]
        input_filepath = '{}/{}_input.png'.format(self.input_dir, sample_token)        
        im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)  
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)      
        
        if self.transforms is not None: 
            augmented = self.transforms(image=im, mask=target)  
            im = augmented['image']

        # in initial version we had extra resize and normalisation    
        im = cv2.resize(im, (self.img_size, self.img_size))        
        im = im.astype(np.float32)/255

        im = torch.from_numpy(im.transpose(2,0,1)) # channels first       

        return im, sample_token  


class BEVMapsTestDataset(torch.utils.data.Dataset):
    """
    Bird-eye-view dataset with maps, test data  

    Args:
        df: Dataframe with sample tokens
        debug: if True, runs the debugging on few images
        img_size: the desired image size to resize to        
        input_dir: directory with inputs and targets (and maps, optionally)
        maps_dir: directory with input maps
        num_classes: numbr of classes in classification head
        transforms: augmentations (albumentations composed list))        
    """         
    def __init__(self, df: pd.DataFrame, 
                 debug: bool, img_size: int, 
                 input_dir: str, maps_dir: str, 
                 num_classes = NUM_CLASSES, transforms = None):
        super(BEVMapsTestDataset, self).__init__()  # inherit it from torch Dataset    
        self.df = df
        self.debug = debug
        self.img_size = img_size
        self.input_dir = input_dir
        self.maps_dir = maps_dir
        self.num_classes = num_classes
        self.transforms = transforms
        
        if self.debug:
            self.df = self.df.head(16)
            print('Debug mode, samples: ', self.df.samples)  
        self.sample_tokens = list(self.df.samples)
        
    def __len__(self):
        return len(self.sample_tokens)
    
    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]
        input_filepath = '{}/{}_input.png'.format(self.input_dir, sample_token)
        map_filepath = '{}/{}_map.png'.format(self.maps_dir, sample_token) 

        im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)  
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)         
        map_im = cv2.imread(map_filepath, cv2.IMREAD_UNCHANGED)
        map_im = cv2.resize(map_im, (768, 768))         
        
        im  = 2*im//3 + map_im//3 # as alternative, we can just add map overlay on image  
        if self.transforms is not None: 
            augmented = self.transforms(image=im, mask=target)  
            im = augmented['image']
            target = augmented['mask']
        
        im = cv2.resize(im, (self.img_size, self.img_size))        
        # print(im.shape, map_im.shape)
        #im = np.concatenate((im, map_im), axis=2)  # second option is concatenate
        #im = im.astype(np.float32)/255       

        im = torch.from_numpy(im.transpose(2,0,1)) # channels first
        
        return im, sample_token   


def visualize_lidar_of_sample(sample_token, axes_limit=80):
    """Helper to visualize sample lidar data"""
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    level5data.render_sample_data(sample_lidar_token, axes_limit=axes_limit)


def plot_img_target(im: torch.Tensor, target: torch.Tensor, sample_token = None, fig_num = 1):
    """Helper to plot imag eand target"""
    im = im.numpy()
    im =np.rint(im*255)
    target = target.numpy()                
    target_as_rgb = np.repeat(target[...,None], 3, 2) # repeat array for three channels

    plt.figure(fig_num, figsize=(16,8))    
    # Transpose the input volume CXY to XYC order, which is what matplotlib requires.
    plt.imshow(np.hstack((im.transpose(1,2,0)[...,:3], target_as_rgb))) 
    plt.title(sample_token)
    plt.show()


def test_dataset_augs(train_df, data_folder):
    """Helper to test data augmentations"""
    train_dataset = BEVImageDataset(fold=0, df=train_df, 
                                     debug=True, img_size=IMG_SIZE, 
                                     input_dir=data_folder, 
                                     transforms = albu_show_transforms)
    for count in range(10):
        # get dataset sample and plot it
        im, target, sample_token = train_dataset[0]
        plot_img_target(im, target, sample_token, fig_num = count+1)
        

def test_maps_dataset():
    """Helper to test dataset with maps"""
    maps_folder = 'C:/Users/New/Documents/Challenges/lyft/input/maps/maps' 
    # dataset
    train_dataset = BEVMapsDataset(df=train_df, 
                                    debug=True, img_size=IMG_SIZE, 
                                    input_dir=data_folder, 
                                    maps_dir=maps_folder,
                                    num_classes = NUM_CLASSES,
                                    transforms = None)
    im, target, labels, sample_token = train_dataset[6]
    plot_img_target(im, target, sample_token, fig_num = 1)
    
    im = im.numpy()
    target = target.numpy()
    # sanity check
    for num in range(0, len(classes)+1):
        print(np.where(target == num))
                                   
                                   

def main():    
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    
    # "bev" folders
    data_folder = os.path.join(OUTPUT_ROOT, "bev_data")
    input_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_input.png")))
    sample_tokens = [x.split("/")[-1].replace("_input.png","") for x in input_filepaths]
    sample_tokens = [x.replace("bev_data\\","") for x in sample_tokens]
    
    # train samples
    df = pd.read_csv('../folds/train_samples.csv')
    print(df.head())
    train_df = df[df['samples'].isin(sample_tokens)]
    print(train_df.head())

    # dataset
    train_dataset = BEVLabelsDataset(df=train_df, 
                                    debug=True, img_size=512, 
                                    input_dir=data_folder,
                                    transforms = albu_show_transforms)
    # get dataset sample and plot it
    im, target, labels, sample_token = train_dataset[10]
    plot_img_target(im, target, sample_token, fig_num = 2)
    
    # sanity check
    im = im.numpy()
    target = target.numpy()
    for num in range(0, len(classes)+1):
        print(np.where(target == num))
  
    test_dataset_augs(train_df, data_folder)
  

if __name__ == '__main__':
    main()
