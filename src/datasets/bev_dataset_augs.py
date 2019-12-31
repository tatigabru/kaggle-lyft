import collections
import glob
import os
import sys
sys.path.append('/home/user/challenges/lyft/lyft_repo/src')

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from skimage.color import label2rgb
from tqdm import tqdm

import albumentations as A
import torch
import torch.utils.data
from configs import (BEV_SHAPE, DATA_ROOT, IMG_SIZE, NUM_CLASSES, OUTPUT_ROOT,
                     PROJECT_ROOT)
from datasets.transforms import (D4_transforms, augment_and_show,
                                 train_transforms, valid_transforms,
                                 visualize_bbox)
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import (Box, LidarPointCloud,
                                                 Quaternion)
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix, view_points




ImageFile.LOAD_TRUNCATED_IMAGES = True


class BevDatasetAugs(torch.utils.data.Dataset):
    """
    Bird-eye-view dataset amended for coco style
    
    Args:
        fold: integer, number of the fold
        df: Dataframe with sample tokens
        level5data: set oj json linked Tables
        debug: if True, runs the debugging on few images
        img_size: the desired image size to resize to        
        input_dir: directory with imputs and targets (and maps, optionally)
        transforms: list of albumentations
        bev_shape: shape of the BEV image
        voxel_size: voxelization paramenters
        z_offset: offset along vertical axis during voxelization 
        if_map: if True, maps are added   
        """    
    def __init__(self, fold: int, df: pd.DataFrame, 
                 level5data,
                 debug: bool, img_size: int, 
                 input_dir: str, transforms = None, 
                 bev_shape = (768, 768, 3),
                 voxel_size = (0.2, 0.2, 1.5), 
                 z_offset = -2,
                 if_map = False):

        super(BevDatasetAugs, self).__init__()  # inherit it from torch Dataset
        self.fold = fold
        self.df = df
        self.level5data = level5data
        self.debug = debug
        self.img_size = img_size
        self.input_dir = input_dir
        self.transforms = transforms
        self.bev_shape = bev_shape
        self.voxel_size = voxel_size
        self.z_offset = z_offset
        self.if_map = if_map
        self.classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    
        if self.debug:
            self.df = self.df.head(16)
            print('Debug mode, samples: ', self.df.samples)  
        self.sample_tokens = list(self.df.samples)

    def __len__(self):
        return len(self.sample_tokens)
    
    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]
        input_filepath = '{}/{}_input.png'.format(self.input_dir, sample_token)        
        # load BEV image
        im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED) 
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)     

        # get annotations
        sample = self.level5data.get("sample", sample_token)
        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = self.level5data.get("sample_data", sample_lidar_token)
        ego_pose = self.level5data.get("ego_pose", lidar_data["ego_pose_token"])
        
        # get boxes from lyft SDK
        boxes = self.level5data.get_boxes(sample_lidar_token)
        move_boxes_to_car_space(boxes, ego_pose)
        # scale_boxes(boxes, box_scale=0.8)

        targets = coco_targets_from_boxes(sample_token, self.bev_shape, 
                                    self.voxel_size, boxes, self.classes, 
                                    self.z_offset)

        # remove small instances
        target = remove_tiny_instances(targets, threshold=1)     
        masks = target['masks']
        boxes = target["boxes"] # format [xmin, ymin, xmax, ymax]
        labels = target["labels"] 
        boxes = list(boxes)    

        # augment image and targets
        if self.transforms is not None:
            bbox_params={'format':'pascal_voc', 'min_area': 5, 'min_visibility': 0.5, 'label_fields': ['category_id']}
            augs = A.Compose(self.transforms, bbox_params=bbox_params, p=1)       
            augmented = augs(image=im, masks=masks, bboxes=boxes, category_id=labels)     
            im = augmented['image']
            masks = augmented['masks']
            boxes = augmented['bboxes']       
                                                    
        # targets to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)  
        im = im.astype(np.float32)         
        im = torch.from_numpy(im.transpose(2,0,1)) # C, H, W

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd  

        im = im.astype(np.float32)/255       
        im = torch.from_numpy(im.transpose(2,0,1)) # channels first

        # numpy image to PIL image
        #im = Image.fromarray(im)        
        return im, target


def car_to_voxel_coords(points, shape, voxel_size, z_offset=0):
    if len(shape) != 3:
        raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")        
    if len(points.shape) != 2 or points.shape[0] not in [3, 4]:
        raise Exception("Input points should be (3,N) or (4,N) in shape, found {}".format(points.shape))

    tm = create_transformation_matrix_to_voxel_space(shape, voxel_size, (0, 0, z_offset))
    p = transform_points(points, tm)
    return p


def create_transformation_matrix_to_voxel_space(shape, voxel_size, offset):
    """
    Constructs a transformation matrix given an output voxel shape such that (0,0,0) ends up in the center.
    Voxel_size defines how large every voxel is in world coordinate, (1,1,1) would be the same as Minecraft voxels.
    
    An offset per axis in world coordinates (metric) can be provided, this is useful for Z (up-down) in lidar points.
    """
    
    shape, voxel_size, offset = np.array(shape), np.array(voxel_size), np.array(offset)
    
    tm = np.eye(4, dtype=np.float32)
    translation = shape/2 + offset/voxel_size
    
    tm = tm * np.array(np.hstack((1/voxel_size, [1])))
    tm[:3, 3] = np.transpose(translation)
    return tm


def transform_points(points, transf_matrix):
    """
    Transform (3,N) or (4,N) points using transformation matrix.
    """
    if points.shape[0] not in [3,4]:
        raise Exception("Points input should be (3,N) or (4,N) shape, received {}".format(points.shape))
    return transf_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]


def move_boxes_to_car_space(boxes, ego_pose):
    """
    Move boxes from world space to car space.
    Note: mutates input boxes.
    """
    translation = -np.array(ego_pose['translation'])
    rotation = Quaternion(ego_pose['rotation']).inverse
    
    for box in boxes:
        # Bring box to car space
        box.translate(translation)
        box.rotate(rotation)


def scale_boxes(boxes, factor):
    """
    Note: mutates input boxes
    """
    for box in boxes:
        box.wlh = box.wlh * factor


def coco_targets_from_boxes(sample_token, bev_shape, voxel_size, boxes, classes, z_offset=-2):
    """Helper to get COCO-style annoations from the SDK lyft data"""
    #print(len(boxes))
    masks = np.zeros((len(boxes), bev_shape[0], bev_shape[0]), dtype=np.uint8)
    labels = []
    coco_boxes = []
    for num, box in enumerate(boxes):
        mask = np.zeros((bev_shape[0], bev_shape[0], 3), dtype=np.uint8)
        # We only use bottom corners for 2D
        corners = box.bottom_corners()        
        corners_voxel = car_to_voxel_coords(corners, bev_shape, voxel_size, z_offset).transpose(1,0)
        corners_voxel = corners_voxel[:,:2] # drop z coord
        corners_voxel = np.clip(corners_voxel, 0, bev_shape[0]-1) # clip boxes coord to the image size
        class_color = classes.index(box.name) + 1 # encode class index in color pixel,starting from 1        
        if class_color == 0:
            raise Exception("Unknown class: {}".format(box.name))
        cv2.drawContours(mask, np.int0([corners_voxel]), 0, (class_color, class_color, class_color), -1)
        mask = mask[:, :, 0] # choose only one channel
        masks[num, :, :] = mask        
        labels.append(class_color)
        # get orthogonal boxes from masks
        pos = np.nonzero(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])             
        coco_boxes.append([xmin, ymin, xmax, ymax])
  
    coco_boxes = np.asarray(coco_boxes, dtype=np.float32)    
    labels = np.asarray(labels, dtype=np.uint8)        
    target = {}
    target["boxes"] = coco_boxes
    target["labels"] = labels
    target["masks"] = masks
    
    return target   


def remove_tiny_instances(targets, threshold=5):
    """Helper to remove tiny instances"""
    labels = targets["labels"]
    boxes = targets["boxes"]
    masks = targets["masks"]
    num_objs = len(labels)
    new_boxes = []
    new_labels = []
    new_masks = []        
    for num in range(num_objs):
        [xmin, ymin, xmax, ymax] = boxes[num, :]
        if abs(xmax - xmin) >= threshold and abs(ymax - ymin) >= threshold:
            new_boxes.append([xmin, ymin, xmax, ymax])
            new_labels.append(labels[num])
            new_masks.append(masks[num, :, :])         
    nmx = np.zeros((len(new_boxes), masks.shape[1], masks.shape[2]), dtype=np.uint8)
    for i, mask in enumerate(new_masks): 
        nmx[i, :, :] = mask
    new_boxes = np.asarray(new_boxes, dtype=np.float32)    
    new_labels = np.asarray(new_labels, dtype=np.uint8)

    new_target = {}
    new_target["boxes"] = new_boxes
    new_target["labels"] = new_labels
    new_target["masks"] = nmx          
    
    return new_target   


def plot_image_mask(im, boxes, masks):
    """Helper, to plot image with boxes and masks"""
    # plot boxes on image
    for box in boxes:
        visualize_bbox(im, box)
        
    # glue masks together
    one_mask = np.zeros_like(masks[0])
    for i, mask in enumerate(masks):
        one_mask += (mask > 0).astype(np.uint8) * i      
    mask_rgb = label2rgb(one_mask, bg_label=0) 

    f, ax = plt.subplots(2, figsize=(16, 16))             
    ax[0].imshow(im)
    ax[0].set_title('Image')        
    ax[1].imshow(mask_rgb, interpolation='nearest')
    ax[1].set_title('Mask')    
    f.tight_layout()
    plt.show()  

    
def main():
    # get data
    level5data = LyftDataset(data_path = '../input/', json_path='../input/train_data', verbose=True) # local laptop
    #level5data = LyftDataset(data_path = '.', json_path='../../input/train_data', verbose=True) # server
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    
    # BEV conversion parameters
    bev_shape = (768, 768, 3)
    voxel_size = (0.2, 0.2, 1.5)
    img_size = 768

    # "bev" folders
    data_folder = os.path.join(OUTPUT_ROOT, "bev_data")
    input_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_input.png")))
    sample_tokens = [x.split("/")[-1].replace("_input.png","") for x in input_filepaths]
    sample_tokens = [x.replace("bev_data\\","") for x in sample_tokens]
    
    # train samples
    df = pd.read_csv('folds/train_samples.csv')
    train_df = df[df['samples'].isin(sample_tokens)]
    print(train_df.head())

    # dataset     
    train_dataset = BevDatasetAugs(fold=0, df=train_df, 
                                    level5data = level5data,
                                    debug=True, 
                                    img_size=bev_shape[0], 
                                    input_dir=data_folder, 
                                    transforms = train_transforms,                                    
                                    bev_shape = bev_shape,
                                    voxel_size = voxel_size, 
                                    z_offset = z_offset)    
    for num in range(5):
        # get dataset sample and plot it
        im, target = train_dataset[9]          
        masks = target['masks'].numpy()
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()    
        plot_image_mask(im, boxes, masks)
     
    # plot insatance sample
    ins_num = 0
    [xmin, ymin, xmax, ymax] = boxes[ins_num, :]
    mask = masks[ins_num, :, :]
     
    plt.figure(1, figsize=(16,8))
    target_as_rgb = np.repeat(mask[...,None], 3, 2)
    # make overlay with the mask
    image = im//2 + target_as_rgb//2
    cv2.rectangle(image,(xmin, ymin),(xmax, ymax),(255,0,255),1)
    plt.imshow(image) 
    plt.title(labels[ins_num])
    plt.show()  


if __name__ == '__main__':   
    main()    
