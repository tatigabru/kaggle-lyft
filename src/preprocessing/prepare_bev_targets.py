import glob
import os
import sys
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
from scipy.spatial.transform import Rotation as RS
from tqdm import tqdm, tqdm_notebook

sys.path.append('/home/user/challenges/lyft/lyft_repo/src')
from configs import BEV_SHAPE, DATA_ROOT, OUTPUT_ROOT
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import (Box, LidarPointCloud,
                                                 Quaternion)
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix, view_points

os.environ["OMP_NUM_THREADS"] = "1"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


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


def test_transform_points():
    # Let's try it with some example values
    tm = create_transformation_matrix_to_voxel_space(shape=(100,100,4), voxel_size=(0.5,0.5,0.5), offset=(0,0,0.5))
    p = transform_points(np.array([[10, 10, 0, 0, 0], [10, 5, 0, 0, 0],[0, 0, 0, 2, 0]], dtype=np.float32), tm)
    print(p)


def car_to_voxel_coords(points, shape, voxel_size, z_offset=0):
    if len(shape) != 3:
        raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")
        
    if len(points.shape) != 2 or points.shape[0] not in [3, 4]:
        raise Exception("Input points should be (3,N) or (4,N) in shape, found {}".format(points.shape))

    tm = create_transformation_matrix_to_voxel_space(shape, voxel_size, (0, 0, z_offset))
    p = transform_points(points, tm)
    return p


def create_voxel_pointcloud(points, shape, voxel_size=(0.5,0.5,1), z_offset=0):

    points_voxel_coords = car_to_voxel_coords(points.copy(), shape, voxel_size, z_offset)
    points_voxel_coords = points_voxel_coords[:3].transpose(1,0)
    points_voxel_coords = np.int0(points_voxel_coords)
    
    bev = np.zeros(shape, dtype=np.float32)
    bev_shape = np.array(shape)

    within_bounds = (np.all(points_voxel_coords >= 0, axis=1) * np.all(points_voxel_coords < bev_shape, axis=1))
    
    points_voxel_coords = points_voxel_coords[within_bounds]
    coord, count = np.unique(points_voxel_coords, axis=0, return_counts=True)
        
    # Note X and Y are flipped:
    bev[coord[:,1], coord[:,0], coord[:,2]] = count
    
    return bev


def normalize_voxel_intensities(bev, max_intensity=16):
    return (bev/max_intensity).clip(0,1)


def get_pointcloud(sample_token, level5data):  
    """Get sample of lidar point cloud
    Transform it to car coordinates
    """
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = level5data.get("sample_data", sample_lidar_token)
    lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)
    lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)

    # sensor (lidar) token
    calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

    # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
    car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                        inverse=False)
    # The lidar pointcloud is defined in the sensor's reference frame.
    # We want it in the car's reference frame, so we transform each point
    lidar_pointcloud.transform(car_from_sensor)

    return lidar_pointcloud


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


def draw_boxes(im, voxel_size, boxes, classes, z_offset=0.0):
    for box in boxes:
        # We only care about the bottom corners
        corners = box.bottom_corners()
        corners_voxel = car_to_voxel_coords(corners, im.shape, voxel_size, z_offset).transpose(1,0)
        corners_voxel = corners_voxel[:,:2] # Drop z coord
        class_color = classes.index(box.name) + 1 # encode class index in color pixel
        print(class_color)
        if class_color == 0:
            raise Exception("Unknown class: {}".format(box.name))
        cv2.drawContours(im, np.int0([corners_voxel]), 0, (class_color, class_color, class_color), -1)

    return im


def test_voxel_img_size(lidar_pointcloud, size_min =0.1, img_size = 768):
    """Test different voxel sizes and image sizes"""
    sizes = np.arange(size_min, size_min*5, 0.1)
    print("sizes: {}".format(sizes))
    z_offset = -2.0
    bev_shape = (img_size, img_size, 3)
    for count, size in enumerate(sizes):
        voxel_size = (size,size,1.5)    
        bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
        # So that the values in the voxels range from 0,1 we set a maximum intensity.
        bev = normalize_voxel_intensities(bev)
        plt.figure(count, figsize=(16,8))
        plt.imshow(bev)
        plt.show()


def plot_bev_and_boxes(sample_token, level5data, size = 0.2, img_size = 768):
    """Plot beird eye view from lidar point cloud
    Plor ground truth"""    
    z_offset = -2.0
    bev_shape = (img_size, img_size, 3)
    voxel_size = (size,size,1.5)  
    lidar_pointcloud = get_pointcloud(sample_token, level5data)  
    bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
    # So that the values in the voxels range from 0,1 we set a maximum intensity.
    bev = normalize_voxel_intensities(bev)

    # get ground truth
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    boxes = level5data.get_boxes(sample_lidar_token)
    lidar_data = level5data.get("sample_data", sample_lidar_token)
    ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
    move_boxes_to_car_space(boxes, ego_pose)
    scale_boxes(boxes, 0.8)

    target_im = np.zeros(bev.shape[:3], dtype=np.uint8)    
    draw_boxes(target_im, voxel_size, boxes, classes, z_offset=z_offset)
    
    plt.figure(1, figsize=(16,8))
    plt.imshow(np.hstack((bev, (target_im > 0).astype(np.float32)))) 
    plt.title(sample_token)

    return bev


def get_semantic_map_around_ego(map_mask, ego_pose, voxel_size=0.2, output_shape=(768, 768)):
    """
    Get map around vehicle, movesit to voxel coordinates and creates an image
    """
    def crop_image(image: np.array,
                           x_px: int,
                           y_px: int,
                           axes_limit_px: int) -> np.array:
                x_min = int(x_px - axes_limit_px)
                x_max = int(x_px + axes_limit_px)
                y_min = int(y_px - axes_limit_px)
                y_max = int(y_px + axes_limit_px)

                cropped_image = image[y_min:y_max, x_min:x_max]

                return cropped_image

    pixel_coords = map_mask.to_pixel_coords(ego_pose['translation'][0], ego_pose['translation'][1])

    extent = voxel_size*output_shape[0]*0.5
    scaled_limit_px = int(extent * (1.0 / (map_mask.resolution)))
    mask_raster = map_mask.mask()

    cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * np.sqrt(2)))

    ypr_rad = Quaternion(ego_pose['rotation']).yaw_pitch_roll
    yaw_deg = -np.degrees(ypr_rad[0])

    rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
    ego_centric_map = crop_image(rotated_cropped, rotated_cropped.shape[1] / 2, rotated_cropped.shape[0] / 2,
                                 scaled_limit_px)[::-1]
    
    ego_centric_map = cv2.resize(ego_centric_map, output_shape[:2], cv2.INTER_NEAREST)

    return ego_centric_map.astype(np.float32)/255


def plot_semantic_map(map_mask, ego_pose):
    """Helperto plot map"""
    ego_centric_map = get_semantic_map_around_ego(map_mask, ego_pose, voxel_size=0.4, output_shape=(336,336)) 
    plt.imshow(ego_centric_map)
    plt.show()


def prepare_training_data_for_scene(first_sample_token, output_folder, level5data, classes, bev_shape, voxel_size, z_offset, box_scale):
    """
    Given a first sample token (in a scene), output rasterized input volumes and targets in birds-eye-view perspective.

    """
    # get the first sample tken for the scene
    sample_token = first_sample_token
    
    while sample_token:   

        sample = level5data.get("sample", sample_token)
        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)

        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])


        global_from_car = transform_matrix(ego_pose['translation'],
                                           Quaternion(ego_pose['rotation']), inverse=False)

        car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                            inverse=False)

        try:
            lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print ("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))
            sample_token = sample["next"]
            continue
        
        bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
        bev = normalize_voxel_intensities(bev)
        
        boxes = level5data.get_boxes(sample_lidar_token)
        target = np.zeros_like(bev)

        move_boxes_to_car_space(boxes, ego_pose)
        scale_boxes(boxes, box_scale)
        target = draw_boxes(target, voxel_size, boxes=boxes, classes=classes, z_offset=z_offset)

        bev_im = np.round(bev*255).astype(np.uint8)
        target_im = target[:,:,0] # take one channel only

        cv2.imwrite(os.path.join(output_folder, "{}_input.png".format(sample_token)), bev_im)
        cv2.imwrite(os.path.join(output_folder, "{}_target.png".format(sample_token)), target_im)
        # go to the next sample token
        sample_token = sample["next"]


def prepare_maps_scene(first_sample_token, output_folder, level5data, map_mask, bev_shape, voxel_size):
    """
    Given a first sample token (in a scene), output rasterized input volumes and targets in birds-eye-view perspective.

    """
    # get the first sample tken for the scene
    sample_token = first_sample_token
    
    while sample_token:   

        sample = level5data.get("sample", sample_token)
        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])        
        
        size_map = bev_shape[0]        
        semantic_im = get_semantic_map_around_ego(map_mask, ego_pose, voxel_size[0], output_shape=(size_map, size_map))
        semantic_im = np.round(semantic_im*255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(output_folder, "{}_map.png".format(sample_token)), semantic_im)
        # go to the next sample token
        sample_token = sample["next"]


def test_prepare_training_data(sample_token, train_data_folder):
    """Opens and plots images saved by prepare_training_data_for_scenefunction""" 
    target_filepath =  f"{train_data_folder}/{sample_token}_target.png"
    input_filepath =  f"{train_data_folder}/{sample_token}_input.png"
    print("target_filepath {}, input_filepath {}".format(target_filepath, input_filepath))

    im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED) 
    target = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)
        
    im = im.astype(np.float32)/255
    im = im.transpose(2,0,1)
    target = (target == 1).astype(np.float32)      
       
    plt.figure(100, figsize=(16,8))
    target_as_rgb = np.repeat(target[...,None], 3, 2)
    # Transpose the input volume CXY to XYC order, which is what matplotlib requires.
    plt.imshow(np.hstack((im.transpose(1,2,0)[...,:3], target_as_rgb))) 
    plt.title(sample_token)
    plt.show()


def prepare_data_pool(df, data_folder, bev_shape, voxel_size, z_offset, box_scale):
    """Prepare input data 
    Args: 
        df: train or val tokens
        data_folder: diractory to save data
        bev_shape: BEV image shape
        voxel_size: size of voxels for voxelization
        z_offset: offset in the vertical direction before voxelization 
        box_scale: rescale of the bounding boxes projections
    """
    NUM_WORKERS = os.cpu_count() 
    print('Number of CPU: ', NUM_WORKERS)

    level5data = LyftDataset(data_path = '.', json_path='../../input/train_data', verbose=True)
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    
    print("Preparing data into {} using {} workers".format(data_folder, NUM_WORKERS))
    first_samples = df.first_sample_token.values
    os.makedirs(data_folder, exist_ok=True)
    
    process_func = partial(prepare_training_data_for_scene,
                           output_folder=data_folder, level5data=level5data, bev_shape=bev_shape, voxel_size=voxel_size, z_offset=z_offset, box_scale=box_scale)

    pool = Pool(NUM_WORKERS)
    for _ in tqdm(pool.imap_unordered(process_func, first_samples), total=len(first_samples)):
        pass
    pool.close()
    del pool


def main():
    # get data
    level5data = LyftDataset(data_path = '.', json_path='../../input/train_data', verbose=True)
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]

    df = pd.read_csv('host_scenes.csv')
    host_count_df = df.groupby("host")['scene_token'].count()
    print(host_count_df)

    sample_token = df.first_sample_token.values[0]
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = level5data.get("sample_data", sample_lidar_token)
    lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)

    # car and sensor coords
    ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
    print("ego_pose: ", ego_pose)
    calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    
    # Homogeneous transformation matrix from car frame to world frame.
    global_from_car = transform_matrix(ego_pose['translation'],
                                    Quaternion(ego_pose['rotation']), inverse=False)

    # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
    car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                        inverse=False)
    lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)

    # The lidar pointcloud is defined in the sensor's reference frame.
    # We want it in the car's reference frame, so we transform each point
    lidar_pointcloud.transform(car_from_sensor)

    # Define hyper parameters 
    #bev_shape = (768, 768, 3)
    #voxel_size = (0.2,0.2,1.5)
    bev_shape = (1024, 1024, 3)
    voxel_size = (0.2,0.2,1.5)

    z_offset = -2.0
    box_scale = 0.8

    #map_mask = level5data.map[0]["mask"]
    #print(map_mask)
    #plot_semantic_map(map_mask, ego_pose)
    #sample_token = train_df.first_sample_token.values[10]
    #bev = plot_bev_and_boxes(sample_token, level5data)

    train_data_folder = os.path.join(OUTPUT_ROOT, "bev_data_1024")
    validation_data_folder = os.path.join(OUTPUT_ROOT, "bev_data_1024")
    os.makedirs(train_data_folder, exist_ok=True)
    os.makedirs(validation_data_folder, exist_ok=True)

    # test on a single scene
    prepare_training_data_for_scene(df.first_sample_token.values[50], 
                                    train_data_folder, 
                                    level5data, 
                                    classes,
                                    bev_shape, voxel_size, z_offset, box_scale)
    test_prepare_training_data(df.first_sample_token.values[50], train_data_folder)

    # get for all scenes
    first_samples = df.first_sample_token.values
    print(first_samples)

    for sample in first_samples:
        print(sample)
        prepare_training_data_for_scene(sample, 
                                    train_data_folder, 
                                    level5data, 
                                    classes,
                                    bev_shape, voxel_size, z_offset, box_scale)
    # multiprocessing option
    #for df, data_folder in [(train_df, train_data_folder), (validation_df, validation_data_folder)]:
    #    prepare_data_pool(df, data_folder, bev_shape, voxel_size, z_offset, box_scale)
    print('Mission accomplished!')


if __name__ == '__main__':   
    main()
