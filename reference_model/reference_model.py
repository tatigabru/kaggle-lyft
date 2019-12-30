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

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

from configs import DATA_ROOT, OUTPUT_ROOT, BEV_SHAPE
os.makedirs(OUTPUT_ROOT, exist_ok=True)


level5data = LyftDataset(data_path = '../input', json_path='../input/train_data', verbose=True)

classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]

def get_host_scenes(level5data, if_save = False):
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

    if if_save:             
        df = pd.DataFrame(entries, columns=["host", "scene_name", "date", "scene_token", "first_sample_token"])
        df.to_csv('host_scenes.csv', index = False)

    return df    

#df = get_host_scenes(level5data, if_save = True)
df = pd.read_csv('host_scenes.csv')
print(df.head())

# train - validation split by car
validation_hosts = ["host-a007", "host-a008", "host-a009"]
validation_df = df[df["host"].isin(validation_hosts)]
vi = validation_df.index
train_df = df[~df.index.isin(vi)]
print(len(train_df), len(validation_df), "train/validation split scene counts")

def get_pointcloud(sample_token):
    """Get one toplidar for one sampkle token"""
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)
    lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
    return lidar_pointcloud

def plot_hyst(lidar_pointcloud, num):
    # A sanity check, the points should be centered around 0 in car space.
    plt.figure(num)
    #plt.hist(lidar_pointcloud.points[0], alpha=0.5, bins=100, label="X")
    #plt.hist(lidar_pointcloud.points[1], alpha=0.5, bins=100, label="Y")
    plt.hist(lidar_pointcloud.points[2], alpha=0.5, bins=100, label="Z")
    plt.legend()
    plt.xlabel("Distance from car along axis")
    plt.ylabel("Amount of points")
    plt.show()

# Creating input and targets
sample_token = train_df.first_sample_token.values[0]
sample = level5data.get("sample", sample_token)

sample_lidar_token = sample["data"]["LIDAR_TOP"]
lidar_data = level5data.get("sample_data", sample_lidar_token)
lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)

ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

# Homogeneous transformation matrix from car frame to world frame.
global_from_car = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)

# Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                   inverse=False)


def matrix_to_voxel_space(shape, voxel_size, offset):
    """
    Voxelize the LIDAR points: we go from a list of coordinates of points, to a X by Y by Z space.
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
    tm = matrix_to_voxel_space(shape=(100,100,4), voxel_size=(0.5,0.5,0.5), offset=(0,0,0.5))
    p = transform_points(np.array([[10, 10, 0, 0, 0], [10, 5, 0, 0, 0],[0, 0, 0, 2, 0]], dtype=np.float32), tm)
    print(p)


def car_to_voxel_coords(points, shape, voxel_size, x_offset = 0, y_offset = 0, z_offset=0):
    if len(shape) != 3:
        raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")
        
    if len(points.shape) != 2 or points.shape[0] not in [3, 4]:
        raise Exception("Input points should be (3,N) or (4,N) in shape, found {}".format(points.shape))

    tm = matrix_to_voxel_space(shape, voxel_size, (x_offset, y_offset, z_offset))
    p = transform_points(points, tm)
    return p


def create_voxel_pointcloud(points, shape, voxel_size=(0.5,0.5,1), x_offset=0, y_offset=0, z_offset=0):
    """Voxelize input points 
       Create BEV array
    """   
    points_voxel_coords = car_to_voxel_coords(points.copy(), shape, voxel_size, x_offset, y_offset, z_offset)
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


def select_pointcloud(points, x_range, y_range, z_range):
    """Select the regoin of the points cloud"""
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[0]
    y_points = points[1] 
    z_points = points[2]

    return sel_points


def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


def point_cloud_2_birdseye(points,
                           res=0.1,
                           side_range=(-150., 150.),  # left-most to right-most
                           fwd_range = (-150., 150.), # back-most to forward-most
                           height_range=(-2., 4.),  # bottom-most to upper-most
                           ):
    """ Creates an 2D birds eye view representation of the point cloud data.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[0]
    y_points = points[1]
    z_points = points[2]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im


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


def draw_boxes(im, voxel_size, boxes, classes, x_offset = 0, y_offset = 0, z_offset=0):
    for box in boxes:
        # We only care about the bottom corners
        corners = box.bottom_corners()
        corners_voxel = car_to_voxel_coords(corners, im.shape, voxel_size, x_offset, y_offset, z_offset).transpose(1,0)
        corners_voxel = corners_voxel[:,:2] # Drop z coord
        class_color = classes.index(box.name) + 1        
        if class_color == 0:
            raise Exception("Unknown class: {}".format(box.name))
        cv2.drawContours(im, np.int0([corners_voxel]), 0, (class_color, class_color, class_color), -1)


def get_pointcloud(sample_token):    
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)
    lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
    return lidar_pointcloud


def plot_hyst(lidar_pointcloud, num):
    # A sanity check, the points should be centered around 0 in car space.
    plt.figure(num)
    #plt.hist(lidar_pointcloud.points[0], alpha=0.5, bins=100, label="X")
    #plt.hist(lidar_pointcloud.points[1], alpha=0.5, bins=100, label="Y")
    plt.hist(lidar_pointcloud.points[2], alpha=0.5, bins=100, label="Z")
    plt.legend()
    plt.xlabel("Distance from car along axis")
    plt.ylabel("Amount of points")
    plt.show()


# hyper parametes
#voxel_size = (0.4,0.4,1.5)
z_offset = -2.0
x_offset = 0
y_offset = 0
bev_shape = (1024, 1024, 3)
size_z = 1.5
sizes = np.arange(0.1, 0.5, 0.1)
size = 0.2
voxel_size = (size,size,size_z)

lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
print(lidar_pointcloud.points.shape)
print(lidar_pointcloud.points.transpose().shape)
# The lidar pointcloud is defined in the sensor's reference frame.
# We want it in the car's reference frame
lidar_pointcloud.transform(car_from_sensor)    


def draw_pointcloud_to_bev(train_df):
    for k, num in enumerate(range(2, 100, 2)):
        sample_token = train_df.first_sample_token.values[num]
        lidar_pointcloud = get_pointcloud(sample_token)
        lidar_pointcloud.transform(car_from_sensor)
        im = point_cloud_2_birdseye(lidar_pointcloud.points, res=0.1)
        plt.figure(k, figsize=(16,8))
        plt.imshow(im, cmap="Spectral", vmin=0, vmax=255)
        plt.show()

# load targets
boxes = level5data.get_boxes(sample_lidar_token)
move_boxes_to_car_space(boxes, ego_pose)
scale_boxes(boxes, 0.8)

for count, x_offset in enumerate(range(-100, 150, 50)):
    for ii, y_offset in enumerate(range(-50, 100, 50)):      
        bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, x_offset = x_offset, y_offset = y_offset, z_offset=z_offset)
        # So that the values in the voxels range from 0,1 we set a maximum intensity.
        bev = normalize_voxel_intensities(bev)
        plt.figure(count+ii+1, figsize=(16,8))
        plt.imshow(bev)
        plt.show()

        target_im = np.zeros(bev.shape[:3], dtype=np.uint8)
        draw_boxes(target_im, voxel_size, boxes, classes, x_offset = x_offset, y_offset = y_offset, z_offset=z_offset)
        
        plt.figure(count + ii+ 40, figsize=(16,8))
        plt.imshow((target_im > 0).astype(np.float32), cmap='Set2')
        plt.show()


def visualize_lidar_of_sample(sample_token, axes_limit=80):
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    level5data.render_sample_data(sample_lidar_token, axes_limit=axes_limit)
    plt.show()
# Don't worry about it being mirrored
visualize_lidar_of_sample(sample_token)

def prepare_training_data_for_scene(first_sample_token, output_folder, bev_shape, voxel_size, z_offset, box_scale):
    """
    Given a first sample token (in a scene), output rasterized input volumes and targets in birds-eye-view perspective.

    """
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
        except Exception as e:
            print ("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))
            sample_token = sample["next"]
            continue  

        boxes = level5data.get_boxes(sample_lidar_token)
        move_boxes_to_car_space(boxes, ego_pose)
        scale_boxes(boxes, box_scale)
        count = 0
        for x_offset in range(-50, 100, 50):
            for y_offset in range(-50, 100, 50):    
                bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, x_offset = x_offset, y_offset = y_offset, z_offset=z_offset)
                bev = normalize_voxel_intensities(bev)                  
                #target = np.zeros_like(bev)
                target = np.zeros(bev.shape[:3], dtype=np.uint8)
                draw_boxes(target, voxel_size, boxes=boxes, classes=classes, x_offset = x_offset, y_offset = y_offset, z_offset=z_offset)            
                print(np.nonzero(target))
                bev_im = np.round(bev*255).astype(np.uint8)
                #target_im = target[:,:,0] # take one channel only
                target_im = np.round(target*255).astype(np.uint8)
                #target_im = (target_im[:,:,0]*255).astype(np.uint8) # take one channel only
                print(np.nonzero(target_im))
                
                # save BEV images
                cv2.imwrite(os.path.join(output_folder, "{}_{}_input.png".format(sample_token, count)), bev_im)
                cv2.imwrite(os.path.join(output_folder, "{}_{}_target.png".format(sample_token, count)), target_im)    
                print('{}_{}_target.png saved'.format(sample_token, count))
                count += 1

        sample_token = sample["next"]


def test_training_sample(sample_token, train_data_folder):
    target_filepath =  f"{train_data_folder}/{sample_token}_0_0_target.png"
    input_filepath =  f"{train_data_folder}/{sample_token}_0_0_input.png"
    print("target_filepath {}, input_filepath {}".format(target_filepath, input_filepath))

    im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED) 
    target = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)
        
    im = im.astype(np.float32)/255
    im = im.transpose(2,0,1)
    target = target.astype(np.int64)       
       
    plt.figure(figsize=(16,8))
    target_as_rgb = np.repeat(target[...,None], 3, 2)
    # Transpose the input volume CXY to XYC order, which is what matplotlib requires.
    plt.imshow(np.hstack((im.transpose(1,2,0)[...,:3], target_as_rgb))) 
    plt.title(sample_token)
    plt.show()


train_data_folder = os.path.join(OUTPUT_ROOT, "bev_train")
validation_data_folder = os.path.join(OUTPUT_ROOT, "bev_val")
os.makedirs(train_data_folder, exist_ok=True)
os.makedirs(validation_data_folder, exist_ok=True)

# Define hyperparameters 
bev_shape = (1024, 1024, 3)
size_z = 1.5
size = 0.2
voxel_size = (size,size,size_z)
z_offset = -2.0
box_scale = 0.8

prepare_training_data_for_scene(df.first_sample_token.values[0], 
                                train_data_folder, 
                                bev_shape, voxel_size, z_offset, box_scale)

test_training_sample(df.first_sample_token.values[0], train_data_folder)

if __name__ == 'main':

    # Define hyperparameters 
    bev_shape = (1024, 1024, 3)
    size_z = 1.5
    size = 0.2
    voxel_size = (size,size,size_z)
    z_offset = -2.0
    box_scale = 0.8

    train_data_folder = os.path.join(OUTPUT_ROOT, "bev_train")
    validation_data_folder = os.path.join(OUTPUT_ROOT, "bev_val")
    os.makedir(train_data_folder, exist_ok=True)
    os.makedir(validation_data_folder, exist_ok=True)

    NUM_WORKERS = os.cpu_count() 
    print(NUM_WORKERS)

    df = pd.read_csv('host_scenes.csv')
    print(df.head())

    prepare_training_data_for_scene(df.first_sample_token.values[0], 
                                    train_data_folder, 
                                    bev_shape, voxel_size, z_offset, box_scale)

    #for df, data_folder in [(train_df, train_data_folder), (validation_df, validation_data_folder)]:
    #    print("Preparing data into {} using {} workers".format(data_folder, NUM_WORKERS))
    #    first_samples = df.first_sample_token.values
           
    #    process_func = partial(prepare_training_data_for_scene,
    #                            output_folder=data_folder, bev_shape=bev_shape, voxel_size=voxel_size, z_offset=z_offset, box_scale=box_scale)
    #    pool = Pool(NUM_WORKERS)
    #    for _ in tqdm(pool.imap_unordered(process_func, first_samples), total=len(first_samples)):
    #        pass
    #    pool.close()
    #    del pool