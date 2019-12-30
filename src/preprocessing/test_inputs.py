from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool

# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.
import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R
import torch

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

from configs import DATA_ROOT, OUTPUT_ROOT, BEV_SHAPE
os.makedirs(OUTPUT_ROOT, exist_ok=True)


level5data = LyftDataset(data_path = '../input', json_path='../input/train_data', verbose=True)

classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]


def test_training_sample(sample_token, train_data_folder):
    target_filepath =  f"{train_data_folder}/{sample_token}_0_0_target.png"
    input_filepath =  f"{train_data_folder}/{sample_token}_0_0_input.png"
    print("target_filepath {}, input_filepath {}".format(target_filepath, input_filepath))

    im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED) 
    target = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)
        
    im = im.astype(np.float32)/255
    target = target.astype(np.int64)        
    im = torch.from_numpy(im.transpose(2,0,1)) # channels first
    target = torch.from_numpy(target)  # single channel

    im = im.numpy()
    target = target.numpy()
    
    plt.figure(figsize=(16,8))
    target_as_rgb = np.repeat(target[...,None], 3, 2)
    # Transpose the input volume CXY to XYC order, which is what matplotlib requires.
    plt.imshow(np.hstack((im.transpose(1,2,0)[...,:3], target_as_rgb))) 
    plt.title(sample_token)
    plt.show()


df = pd.read_csv('host_scenes.csv')
print(df.head())
train_data_folder = os.path.join(OUTPUT_ROOT, "bev_train")
os.makedirs(train_data_folder, exist_ok=True)

test_training_sample(df.first_sample_token.values[0], train_data_folder)
