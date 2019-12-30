from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool
import matplotlib.pyplot as plt
# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.
import os
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm, tqdm_notebook
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R
import time
import gc

# lyft SDK imports
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from lyft_dataset_sdk.utils.map_mask import MapMask
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer

# current project imports 
from unet import UNet
from datasets.BEVDataset import BEVTestDataset
from datasets.lyft_test_dataset import LyftTestDataset
from configs import OUTPUT_ROOT, IMG_SIZE, NUM_CLASSES
from models import fpn_resnet50, get_smp_model
from transforms import (train_transform, test_transform, tensor_transform,
                        crop_d4_transforms, albu_valid_tansforms, albu_test_tansforms)
from train import set_seed, visualize_predictions


def visualize_lidar_of_sample(sample_token, axes_limit=80):
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    level5data.render_sample_data(sample_lidar_token, axes_limit=axes_limit)


def calc_detection_box(prediction_opened, class_probability):

    sample_boxes = []
    sample_detection_scores = []
    sample_detection_classes = []
    
    contours, hierarchy = cv2.findContours(prediction_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        
        # Let's take the center pixel value as the confidence value
        box_center_index = np.int0(np.median(box, axis=0))
        
        for class_index in range(len(classes)):
            box_center_value = class_probability[class_index+1, box_center_index[1], box_center_index[0]]
            
            # Let's remove candidates with very low probability
            if box_center_value < 0.3:
                continue
            
            box_center_class = classes[class_index]

            box_detection_score = box_center_value
            sample_detection_classes.append(box_center_class)
            sample_detection_scores.append(box_detection_score)
            sample_boxes.append(box)
            
    return np.array(sample_boxes),sample_detection_scores,sample_detection_classes
    

def open_preds(predictions_non_class0):

    predictions_opened = np.zeros((predictions_non_class0.shape), dtype=np.uint8)

    for i, p in enumerate(tqdm(predictions_non_class0)):
        thresholded_p = (p > background_threshold).astype(np.uint8)
        predictions_opened[i] = cv2.morphologyEx(thresholded_p, cv2.MORPH_OPEN, kernel)
        
    return predictions_opened


def load_model(model, checkpoint_path: str):
    """Loads model weigths, epoch for inference
    """  
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    print('Loaded model from {}, epoch {}'.format(checkpoint_path, epoch))


def inference(model, model_name: str, data_folder: str, fold: int, 
              debug=False, img_size=IMG_SIZE,
              batch_size = 8, num_workers=4):
    """
    Model inference
    
    Input: 
        model : PyTorch model
        model_name : string name for model for checkpoints saving
        fold: evaluation fold number, 0-3
        debug: if True, runs the debugging on few images 
        img_size: size of images for training (for pregressive learning)
        batch_size: number of images in batch
        num_workers: number of workers available
        resume_weights: directory with weights to resume (if avaialable)         
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
        
    # We weight the loss for the 0 class lower to account for (some of) the big class imbalance
    class_weights = torch.from_numpy(np.array([0.2] + [1.0]*NUM_CLASSES, dtype=np.float32))
    class_weights = class_weights.to(device)

    # choose test samples
    input_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_input.png")))
    sample_tokens = [x.split("/")[-1].replace("_input.png","") for x in input_filepaths]
    sample_tokens = [x.replace("bev_data\\","") for x in sample_tokens] 
    
    #creates directories for test predictions checkpoints, tensorboard and predicitons      
    predictions_dir  = f'{OUTPUT_ROOT}/test_preds/{model_name}_fold_{fold}'
    test_outputs_dir = f'{OUTPUT_ROOT}/test_outs/{model_name}_fold_{fold}'    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(test_outputs_dir, exist_ok=True)
    
    test_dataset = BEVTestDataset(sample_tokens=sample_tokens, 
                                  debug=debug, img_size=img_size, 
                                  input_dir=data_folder,
                                  transforms = albu_test_tansforms) 

    # dataloaders for test    
    dataloader_test = DataLoader(test_dataset,
                                 num_workers=num_workers,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=os.cpu_count() * 2)
    print('{} test images'.format(len(test_dataset)))   
    
    # We perform an opening morphological operation to filter tiny detections
    # Note that this may be problematic for classes that are inherently small (e.g. pedestrians)..
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gc.collect()

    progress_bar = tqdm(test_loader)
    # We quantize to uint8 here to conserve memory. We're allocating >20GB of memory otherwise.
    predictions = np.zeros((len(test_loader), 1+len(classes), img_size, img_size), dtype=np.uint8)    
    sample_tokens, all_losses = [], []
    detection_boxes, detection_scores, detection_classes = [], [], []

    # Arbitrary threshold in our system to create a binary image to fit boxes around.
    background_threshold = 200
    with torch.no_grad():
        model.eval()
        for ii, (X, batch_sample_tokens) in enumerate(progress_bar):
            sample_tokens.extend(batch_sample_tokens)

            X = X.to(device)  # [N, 1, H, W]
            prediction = model(X)  # [N, 2, H, W]

            prediction = F.softmax(prediction, dim=1)

            prediction_cpu = prediction.cpu().numpy()
            predictions = np.round(prediction_cpu * 255).astype(np.uint8)

            # Get probabilities for non-background
            predictions_non_class0 = 255 - predictions[:, 0]
            predictions_opened = np.zeros((predictions_non_class0.shape), dtype=np.uint8)

            for i, p in enumerate(predictions_non_class0):
                thresholded_p = (p > background_threshold).astype(np.uint8)
                predictions_opened[i] = cv2.morphologyEx(thresholded_p, cv2.MORPH_OPEN, kernel)

                sample_boxes, sample_detection_scores, sample_detection_classes = calc_detection_box(predictions_opened[i],
                                                                                                    predictions[i])

                detection_boxes.append(np.array(sample_boxes))
                detection_scores.append(sample_detection_scores)
                detection_classes.append(sample_detection_classes)
    
    print("Total amount of boxes:", np.sum([len(x) for x in detection_boxes]))        
    ind = 11
    # Visualize the boxes in the first sample
    t = np.zeros_like(predictions_opened[0])
    for sample_boxes in detection_boxes[ind]:
        box_pix = np.int0(sample_boxes)
        cv2.drawContours(t, [box_pix], 0, (255), 2)


def create_submit():    
    pred_box3ds = []

    # This could use some refactoring..
    for (sample_token, sample_boxes, sample_detection_scores, sample_detection_class) in tqdm_notebook(
            zip(sample_tokens, detection_boxes, detection_scores, detection_classes), total=len(sample_tokens)):
        sample_boxes = sample_boxes.reshape(-1, 2)  # (N, 4, 2) -> (N*4, 2)
        sample_boxes = sample_boxes.transpose(1, 0)  # (N*4, 2) -> (2, N*4)

        # Add Z dimension
        sample_boxes = np.vstack((sample_boxes, np.zeros(sample_boxes.shape[1]),))  # (2, N*4) -> (3, N*4)

        sample = level5data.get("sample", sample_token)
        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)
        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        ego_translation = np.array(ego_pose['translation'])

        global_from_car = transform_matrix(ego_pose['translation'],
                                           Quaternion(ego_pose['rotation']), inverse=False)

        car_from_voxel = np.linalg.inv(create_transformation_matrix_to_voxel_space(bev_shape, voxel_size, (0, 0, z_offset)))
        global_from_voxel = np.dot(global_from_car, car_from_voxel)
        sample_boxes = transform_points(sample_boxes, global_from_voxel)

        # We don't know at where the boxes are in the scene on the z-axis (up-down), let's assume all of them are at
        # the same height as the ego vehicle.
        sample_boxes[2, :] = ego_pose["translation"][2]

        # (3, N*4) -> (N, 4, 3)
        sample_boxes = sample_boxes.transpose(1, 0).reshape(-1, 4, 3)

        # empirical heights
        box_height = np.array([class_heights[cls] for cls in sample_detection_class])

        # To get the center of the box in 3D, we'll have to add half the height to it.
        sample_boxes_centers = sample_boxes.mean(axis=1)
        sample_boxes_centers[:, 2] += box_height / 2

        # Width and height is arbitrary - we don't know what way the vehicles are pointing from our prediction segmentation
        # It doesn't matter for evaluation, so no need to worry about that here.
        # Note: We scaled our targets to be 0.8 the actual size, we need to adjust for that
        sample_lengths = np.linalg.norm(sample_boxes[:, 0, :] - sample_boxes[:, 1, :], axis=1) * 1 / box_scale
        sample_widths = np.linalg.norm(sample_boxes[:, 1, :] - sample_boxes[:, 2, :], axis=1) * 1 / box_scale

        sample_boxes_dimensions = np.zeros_like(sample_boxes_centers)
        sample_boxes_dimensions[:, 0] = sample_widths
        sample_boxes_dimensions[:, 1] = sample_lengths
        sample_boxes_dimensions[:, 2] = box_height

        for i in range(len(sample_boxes)):
            translation = sample_boxes_centers[i]
            size = sample_boxes_dimensions[i]
            class_name = sample_detection_class[i]
            ego_distance = float(np.linalg.norm(ego_translation - translation))

            # Determine the rotation of the box
            v = (sample_boxes[i, 0] - sample_boxes[i, 1])
            v /= np.linalg.norm(v)
            r = R.from_dcm([
                [v[0], -v[1], 0],
                [v[1], v[0], 0],
                [0, 0, 1],
            ])
            quat = r.as_quat()
            # XYZW -> WXYZ order of elements
            quat = quat[[3, 0, 1, 2]]

            detection_score = float(sample_detection_scores[i])

            box3d = Box3D(
                sample_token=sample_token,
                translation=list(translation),
                size=list(size),
                rotation=list(quat),
                name=class_name,
                score=detection_score
            )
            pred_box3ds.append(box3d)
    sub = {}
    for i in tqdm_notebook(range(len(pred_box3ds))):
        #     yaw = -np.arctan2(pred_box3ds[i].rotation[2], pred_box3ds[i].rotation[0])
        yaw = 2 * np.arccos(pred_box3ds[i].rotation[0]);
        pred = str(pred_box3ds[i].score / 255) + ' ' + str(pred_box3ds[i].center_x) + ' ' + \
            str(pred_box3ds[i].center_y) + ' ' + str(pred_box3ds[i].center_z) + ' ' + \
            str(pred_box3ds[i].width) + ' ' \
            + str(pred_box3ds[i].length) + ' ' + str(pred_box3ds[i].height) + ' ' + str(yaw) + ' ' \
            + str(pred_box3ds[i].name) + ' '

        if pred_box3ds[i].sample_token in sub.keys():
            sub[pred_box3ds[i].sample_token] += pred
        else:
            sub[pred_box3ds[i].sample_token] = pred

    sample_sub = pd.read_csv(base_path + 'sample_submission.csv')
    for token in set(sample_sub.Id.values).difference(sub.keys()):
        print(token)
        sub[token] = ''

    sub = pd.DataFrame(list(sub.items()))
    sub.columns = sample_sub.columns
    sub.head()
    sub.tail()
    sub.to_csv(f'{submit_name}.csv', index=False)


def main():

    set_seed(seed=1234)
    NUM_WORKERS = os.cpu_count() * 3
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
        
    #train_dataset = LyftDataset(data_path='.', json_path='../input/3d-object-detection-for-autonomous-vehicles/train_data', verbose=True)
    #level5data = LyftTestDataset(data_path='.', json_path='../input/3d-object-detection-for-autonomous-vehicles/test_data', verbose=True)
   
    class_heights = {'animal':0.51,'bicycle':1.44,'bus':3.44,'car':1.72,'emergency_vehicle':2.39,'motorcycle':1.59,
                'other_vehicle':3.23,'pedestrian':1.78,'truck':3.44}    

    # load data
    data_folder = os.path.join(OUTPUT_ROOT, "bev_test_1024")
    # choose test samples
    input_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_input.png")))
    sample_tokens = [x.split("/")[-1].replace("_input.png","") for x in input_filepaths]
    sample_tokens = [x.replace("bev_data\\","") for x in sample_tokens] 
    df = pd.read_csv('folds/test_host_scenes.csv')
    print(df.head())
   
    # model 
    model = get_smp_model(encoder='resnext101', num_classes=len(classes)+1)
    # load model checkpoint  
    checkpoint= f'{OUTPUT_ROOT}/checkpoints/unet_4_32_768_fold_3/unet_4_32_1024_fold_3_epoch_15.pth'   
    load_model(model, checkpoint)
    model = model.to(device)
    model.eval()

    test_dataset = BEVTestDataset(sample_tokens = samples_test, 
                                 debug=True, img_size=IMG_SIZE, 
                                 input_dir: str, transforms = albu_test_tansforms)
    im, sample_token = test_dataset[1]
    im = im.numpy()

    plt.figure(figsize=(16,8))
    # Transpose the input volume CXY to XYC order, which is what matplotlib requires.
    plt.imshow(im.transpose(1,2,0)[...,:3])
    plt.title(sample_token)
    plt.show()

    visualize_lidar_of_sample(sample_token)

    box_scale = 0.8


if __name__ == '__main__':
    main()
