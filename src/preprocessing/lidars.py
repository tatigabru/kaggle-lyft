import json
import os.path
import math
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from pyquaternion import Quaternion
from sklearn.model_selection import train_test_split
from configs import DATA_ROOT

# Load the dataset
DATA_DIR = '../input/'

class Table:
    def __init__(self, data):
        self.data = data
        self.index = {x['token']: x for x in data}

def load_table(name, root=os.path.join(DATA_DIR, 'train_data')):
    with open(os.path.join(root, name), 'rb') as f:
        return Table(json.load(f))

#scene - 25-45 seconds snippet of a car's journey.
#sample - An annotated snapshot of a scene at a particular timestamp.
#sample_data - Data collected from a particular sensor.
#sample_annotation - An annotated instance of an object within our interest.
#instance - Enumeration of all object instance we observed.
#category - Taxonomy of object categories (e.g. vehicle, human).
#calibrated sensor - Definition of a particular sensor as calibrated on a particular vehicle.
#ego_pose - Ego vehicle poses at a particular timestamp.
#map - Map data that is stored as binary semantic masks from a top-down view.    
    
scene = load_table('scene.json')
sample = load_table('sample.json')
sample_data = load_table('sample_data.json')
ego_pose = load_table('ego_pose.json')
calibrated_sensor = load_table('calibrated_sensor.json')
my_scene = scene.data[0]
print(my_scene)


def get_scene_tokens(scene: Table, if_save = False):
    """Get all scene tokens"""
    my_scene = scene.data[0]
    scene_tokens = []
    for num in range(180):
        my_scene = scene.data[num]
        token = my_scene['token']
        scene_tokens.append(token)    
    #print(scene_tokens)
    if if_save:
        tokens_df = pd.DataFrame(scene_tokens, columns=["scenes"])
        tokens_df.to_csv('scene_tokens.csv', index=False)
    return scene_tokens

scene_tokens = get_scene_tokens(scene, if_save = True)

def get_scene_samples(scene: Table, if_save = False):
    """Get all samples in scenes dictionary"""
    my_scene = scene.data[0]
    all_tokens = {}
    # loop over scenes
    for num in range(180):
        my_scene = scene.data[num]
        token = my_scene['token']  
        # get all 126 samples tokens for this scene
        sample_token = my_scene["first_sample_token"]
        scene_sample_tokens = []
        scene_sample_tokens.append(sample_token)
        for num in range(125):
            next_token = sample.index[sample_token]["next"] # proceed to next sample
            #print(next_token)
            scene_sample_tokens.append(next_token)
            sample.index[next_token]
            sample_token = next_token
        all_tokens[token] = scene_sample_tokens    
    #print(all_tokens)    
    if if_save: 
        all_tokens_df = pd.DataFrame.from_dict(all_tokens).to_csv('scenes_samples.csv', index=False)  
    return all_tokens    

all_tokens = get_scene_samples(scene, if_save = False)  

def get_all_samples(scene: Table, if_save = False):
    """Get all samples for all scenes"""
    scene_sample_tokens = []
    for num in range(178):
        my_scene = scene.data[num]
        # get all 126 samples tokens for this scene
        sample_token = my_scene["first_sample_token"] 
        scene_sample_tokens.append(sample_token)   
        for num in range(125):
            next_token = sample.index[sample_token]["next"] # proceed to next sample
            #print(next_token)
            scene_sample_tokens.append(next_token)
            sample.index[next_token]
            sample_token = next_token
    if if_save:    
        print(scene_sample_tokens)    
        samples_df = pd.DataFrame(scene_sample_tokens, columns=["scenes"])
        samples_df.to_csv('samples.csv', index=False)
    return scene_sample_tokens   

#sample_tokens = get_all_samples(scene, if_save = False)
#print(sample_tokens)

# split scenes
scenes_train, scenes_val = train_test_split(
                            scene_tokens, test_size=0.33, random_state=413)
print('train scenes {}, val scenes {}'.format(len(scenes_train), len(scenes_val)))

# get samples tokens in tain and val scenes
train_tokens = [all_tokens[scenes_train[x]] for x in range(len(scenes_train))]
train_tokens = np.array(train_tokens)
train_tokens = np.concatenate(train_tokens)

val_tokens = [all_tokens[scenes_val[x]] for x in range(len(scenes_val))]
val_tokens = np.array(val_tokens)
val_tokens = np.concatenate(val_tokens)
#print(train_tokens.shape, val_tokens.shape)


# for debugging we'll get only samples from the first training scene.
sample_tokens = all_tokens[scenes_train[0]]
sample_token = sample_tokens[0]
print(sample_token)

def get_sample_lidar(sample_token: str):
    """Get lidar data for a sample
        Input: sample_token: string    
        Output: a numpy array of x,y,z lidar point cloud
    """
    # get all data for the same lidar1 only
    for x in sample_data.data:
        if x['sample_token'] == sample_token and 'lidar1' in x['filename']:
            lidar = x
    lidar_data = np.fromfile(os.path.join(DATA_DIR, lidar['filename']).replace('/lidar/', '/train_lidar/'), dtype=np.float32).reshape(-1, 5)[:, :3]     
    return lidar, lidar_data

"""
Coordinates transforms

"""
# The lidar pointcloud is defined in the sensor's reference frame.
# We want it in the car's reference frame, so we transform each point
def rotate_points(points, rotation, inverse=False):
    assert points.shape[1] == 3
    q = Quaternion(rotation)
    if inverse:
        q = q.inverse
    return np.dot(q.rotation_matrix, points.T).T
    
def apply_pose(points, cs):
    """ Translate (lidar) points to vehicle coordinates, given a calibrated sensor.
    """
    points = rotate_points(points, cs['rotation'])
    points = points + np.array(cs['translation'])
    return points

def inverse_apply_pose(points, cs):
    """ Reverse of apply_pose (we'll need it later).
    """
    points = points - np.array(cs['translation'])
    points = rotate_points(points, np.array(cs['rotation']), inverse=True)
    return points

def car_to_global(lidar, lidar_data):
    """From car frame to world frame"""
    ep = ego_pose.index[lidar["ego_pose_token"]]
    lidar_data = apply_pose(lidar_data, ep)
    return lidar_data 

def sensor_to_car(lidar, lidar_data):
    """sensor coordinate frame to ego car frame"""
    cs = calibrated_sensor.index[lidar['calibrated_sensor_token']] 
    lidar_data = apply_pose(lidar_data, cs)
    return lidar_data
   
def matrix_to_voxel_space(shape, voxel_size, offset):
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

def car_to_voxel_coords(points, shape, voxel_size, z_offset=0):
    if len(shape) != 3:
        raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")
        
    if len(points.shape) != 2 or points.shape[0] not in [3, 4]:
        raise Exception("Input points should be (3,N) or (4,N) in shape, found {}".format(points.shape))

    tm = matrix_to_voxel_space(shape, voxel_size, (0, 0, z_offset))
    p = transform_points(points, tm)
    return p


sample_token = sample_tokens[0]
lidar, lidar_data = get_sample_lidar(sample_token)
print('lidar_data.shape {}'.format(lidar_data.shape))
print('lidar data: ', lidar)

# labels
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv')).set_index('Id')
train_df.loc[sample_token]

def get_annotations(train_df: pd.DataFrame, sample_token: str):
    annotations = np.array(train_df.loc[sample_token].PredictionString.split()).reshape(-1, 8)
    return {
        'point': annotations[:, :3].astype(np.float32),
        'wlh': annotations[:, 3:6].astype(np.float32),
        'rotation': annotations[:, 6].astype(np.float32),
        'cls': np.array(annotations[:, 7]),
    }

print('anns: ', get_annotations(train_df, sample_token))


def viz_lidar(lidar, lidar_data, clip=50, skip_apply_pose=False, fig_num = 1):    
    cs = calibrated_sensor.index[lidar['calibrated_sensor_token']] 
    color = [0, 0, 1, 0.5]
    all_points = []
    if not skip_apply_pose:
        lidar_data = apply_pose(lidar_data, cs)        
    # plot PointCloud    
    plt.figure(fig_num, figsize=(12, 12))
    plt.axis('equal')
    plt.grid()   
    plt.scatter(np.clip(lidar_data[:, 0], -clip, clip), np.clip(lidar_data[:, 2], -clip, clip), s=1, c=color)


def viz_annotation_centers(train_df: pd.DataFrame, sample_token:str, lidar:Table, clip=50):
    # translate annotation points to the car frame
    ego_pose_token = lidar['ego_pose_token'] 
    ep = ego_pose.index[ego_pose_token]
    annotations = get_annotations(train_df, sample_token)
    car_points = annotations['point'][annotations['cls'] == 'car']
    car_points = inverse_apply_pose(car_points, ep)    
    plt.scatter(np.clip(car_points[:, 0], -clip, clip),
                np.clip(car_points[:, 1], -clip, clip),
                s=30,
                color='black')
    
viz_lidar(lidar, lidar_data, clip=200, fig_num = 1)
viz_lidar(lidar, lidar_data, clip=20, fig_num = 2)
viz_annotation_centers(train_df, sample_token, lidar, clip=20)

def plot_lidar_distribution(lidar, lidar_data, if_apply_pose=True):    
    cs = calibrated_sensor.index[lidar['calibrated_sensor_token']]   
    if if_apply_pose:
        lidar_data = apply_pose(lidar_data, cs)     
    # A sanity check, the points should be centered around 0 in car space.
    plt.hist(lidar_data[:, 0], alpha=0.5, bins=30, label="X")
    plt.hist(lidar_data[:, 0], alpha=0.5, bins=30, label="Y")
    plt.hist(lidar_data[:, 2], alpha=0.5, bins=30, label="Z")
    plt.xlim(-20, 20)
    plt.legend()
    plt.xlabel("Distance from car along axis")
    plt.ylabel("Amount of points")
    plt.show()

#plot_lidar_distribution(lidar, lidar_data, if_apply_pose=True)    

def create_voxel_pointcloud(points, shape, voxel_size=(0.5,0.5,1), z_offset=0):
    """Voxelize lidar points
       Create Bird-eye-view projection
    """   
    points_voxel_coords = car_to_voxel_coords(points.copy(), shape, voxel_size, z_offset)
    points_voxel_coords = points_voxel_coords[:3].transpose(1,0)
    points_voxel_coords = np.int0(points_voxel_coords)
    
    bev = np.zeros(shape, dtype=np.float32)
    bev_shape = np.array(shape)

    within_bounds = (np.all(points_voxel_coords >= 0, axis=1) * np.all(points_voxel_coords < bev_shape, axis=1))
    
    points_voxel_coords = points_voxel_coords[within_bounds]
    coord, count = np.unique(points_voxel_coords, axis=0, return_counts=True)
     
    bev[coord[:,0], coord[:,1], coord[:,2]] = count    
    return bev

def normalize_voxel_intensities(bev, max_intensity=16):
    return (bev/max_intensity).clip(0,1)

def rotate(origin, point, angle):
    ox, oy, _ = origin
    px, py, pz = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy, pz]


def make_box_coords(center, wlh, rotation):
    """Make box coordinates from anns"""
    planar_wlh = copy.deepcopy(wlh)
    planar_wlh = planar_wlh[[1,0,2]]
    bottom_center = copy.deepcopy(center)
    bottom_center[-1] = bottom_center[-1] - planar_wlh[-1] / 2

    bottom_points = []
    bottom_points.append(bottom_center + planar_wlh * [1, 1, 0] / 2)
    bottom_points.append(bottom_center + planar_wlh * [-1, -1, 0] / 2)
    bottom_points.append(bottom_center + planar_wlh * [1, -1, 0] / 2)
    bottom_points.append(bottom_center + planar_wlh * [-1, 1, 0] / 2)
    bottom_points = np.array(bottom_points)

    rotated_bottom_points = []
    for point in bottom_points:
        rotated_bottom_points.append(rotate(bottom_center, point, rotation))

    rotated_bottom_points = np.array(rotated_bottom_points)
    rotated_top_points = rotated_bottom_points + planar_wlh * [0,0,1]
    box_points = np.concatenate([rotated_bottom_points, rotated_top_points], axis=0)
       
    return box_points


def get_boxes(train_df: pd.DataFrame, sample_token:str, class_name = 'car'):
    """Get boxes withoug SDK"""
    # translate annotation points to the car frame
    annotations = get_annotations(train_df, sample_token)
    cls_centers = annotations['point'][annotations['cls'] == class_name]
    cls_wlhs = annotations['wlh'][annotations['cls'] == class_name]
    cls_rotations = annotations['rotation'][annotations['cls'] == class_name]
    # move boxes from global to car
    lidar, _ = get_sample_lidar(sample_token)
    ego_pose_token = lidar['ego_pose_token'] 
    ep = ego_pose.index[ego_pose_token]
    
    all_boxes = []
    for k in range(len(cls_centers)):
        center = cls_centers[k]
        wlh = cls_wlhs[k]
        rotation = cls_rotations[k]
        box_points = make_box_coords(center, wlh, rotation)
        box_points = inverse_apply_pose(box_points, ep)
        all_boxes.append(box_points)
    all_boxes = np.array(all_boxes)    
    return all_boxes 
    
def select_pointcloud_crop(lidar, lidar_data):

    return points


# hyper parameters
voxel_size = (0.5, 0.5, 1.5)
z_offset = -2.0
bev_shape = (512, 512, 3)

bev = create_voxel_pointcloud(lidar_data.transpose(), bev_shape, voxel_size=voxel_size, z_offset=z_offset)
bev = normalize_voxel_intensities(bev)

plt.figure(3, figsize=(16,8))
plt.imshow(bev)
plt.show()    

boxes = get_boxes(sample_lidar_token)
target_im = np.zeros(bev.shape[:3], dtype=np.uint8)


def scale_boxes(boxes, factor):
    """
    Note: mutates input boxes
    """
    for box in boxes:
        box.wlh = box.wlh * factor
​
def draw_boxes(im, voxel_size, boxes, classes, z_offset=0.0):
    for box in boxes:
        # We only care about the bottom corners
        corners = box.bottom_corners()
        corners_voxel = car_to_voxel_coords(corners, im.shape, voxel_size, z_offset).transpose(1,0)
        corners_voxel = corners_voxel[:,:2] # Drop z coord
​
        class_color = classes.index(box.name) + 1
        
        if class_color == 0:
            raise Exception("Unknown class: {}".format(box.name))
​
        cv2.drawContours(im, np.int0([corners_voxel]), 0, (class_color, class_color, class_color), -1)
​
​
​
#move_boxes_to_car_space(boxes, ego_pose)
#scale_boxes(boxes, 0.8)
#draw_boxes(target_im, voxel_size, boxes, classes, z_offset=z_offset)
​
#plt.figure(figsize=(8,8))
#plt.imshow((target_im > 0).astype(np.float32), cmap='Set2')
#plt.show()



# Don't worry about it being mirrored.
visualize_lidar_of_sample(sample_token)