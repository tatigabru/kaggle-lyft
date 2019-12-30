"""
Create Lyft Dataset

use KITTI Dataset as an example 
https://github.com/sshaoshuai/PointRCNN/blob/master/lib/datasets/kitti_dataset.py

dataset.py
"""

import os
import numpy as np
from torch.utils.data import Dataset
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
from PIL import Image
import json
import os.path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

from configs import DATA_DIR

# in configs.py we have 
ON_KAGGLE = False
DATA_DIR = '/kaggle/input/3d-object-detection-for-autonomous-vehicles/' if ON_KAGGLE else '../input/'
OUTPUT_DIR = '../output/'
# load data
level5data = LyftDataset(data_path=DATA_DIR, json_path='train_data', verbose=True)
# output
os.makedirs(OUTPUT_DIR, exist_ok=True)

classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]

records = [(level5data.get('sample', record['first_sample_token'])['timestamp'], record) for record in level5data.scene]
entries = []

for start_time, record in sorted(records):
    start_time = level5data.get('sample', record['first_sample_token'])['timestamp'] / 1000000
    token = record['token']
    name = record['name']
    date = datetime.utcfromtimestamp(start_time)
    host = "-".join(record['name'].split("-")[:2])
    first_sample_token = record["first_sample_token"]
    entries.append((host, name, date, token, first_sample_token))
            
df = pd.DataFrame(entries, columns=["host", "scene_name", "date", "scene_token", "first_sample_token"])
df.to_csv('host_scenes.csv', index = False)




class CarsDataset(Dataset):

    def __init__(self, root_dir = DATA_DIR, split='train'):
        self.split = split
        self.root_dir = root_dir
        is_test = self.split == 'test'
        self.string_dir = root_dir + 'test_' if is_test else root_dir + 'train')

        split_dir = os.path.join(root_dir, 'ImageSets', split + '.txt')
        
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_sample = self.image_idx_list.__len__()

        self.image_dir = os.path.join(self.root_dir, 'train_images')
        self.lidar_dir = os.path.join(self.root_dir, 'train_lidar')
        self.data_dir  = os.path.join(self.root_dir, 'train_data')
        self.maps_dir  = os.path.join(self.root_dir, 'train_maps')  
        

    def get_image(self, idx):
        assert False, 'DO NOT USE cv2 NOW, AVOID DEADLOCK'
        import cv2
        # cv2.setNumThreads(0)  # for solving deadlock when switching epoch
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, '%06d.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError