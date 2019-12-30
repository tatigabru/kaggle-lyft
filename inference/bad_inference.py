import glob
import os
from datetime import datetime
import albumentations as A
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm, tqdm_notebook
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
import time
from pathlib import Path
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
import torch
import torch.nn.functional as F
import torch.utils.data

base_path = '3d-object-detection-for-autonomous-vehicles/'

json_path = base_path + 'test_data'
level5data = LyftDataset(data_path=base_path + 'LyftDataset_test', json_path=json_path, verbose=True)

classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
NUM_CLASSES = 9

voxel_size = (0.4, 0.4, 1.5)
z_offset = -2.0
bev_shape = (336 * 2, 2 * 336, 3)
ARTIFACTS_FOLDER = base_path + "bev_test"
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)
test_data_folder = os.path.join(ARTIFACTS_FOLDER, "bev_test_data")


def get_unet_model(in_channels=3, num_output_classes=2):
    model = UNet(in_channels=in_channels, n_classes=num_output_classes, wf=5, depth=4, padding=True, up_mode='upsample')
    return model


albu_test_tansforms = A.Compose([  # no augmentations
    A.Normalize(),
])
# We weigh the loss for the 0 class lower to account for (some of) the big class imbalance.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_weights = torch.from_numpy(np.array([0.2] + [1.0] * len(classes), dtype=np.float32))
class_weights = class_weights.to(device)
batch_size = 4
model = get_unet_model(num_output_classes=len(classes) + 1)
# model = get_smp_model(encoder='resnet152', num_classes=len(classes)+1)
state = torch.load('runs/Nov04_17_39_07_unet/checkpoints/train.1.pth')['model_state_dict']
model.load_state_dict(state)
model = model.to(device)
model.eval()


# %%
class LyftTestDataset(LyftDataset):
    """Database class for Lyft Dataset to help query and retrieve information from the database."""

    def __init__(self, data_path: str, json_path: str, verbose: bool = True, map_resolution: float = 0.1):
        """Loads database and creates reverse indexes and shortcuts.
        Args:
            data_path: Path to the tables and data.
            json_path: Path to the folder with json files
            verbose: Whether to print status messages during load.
            map_resolution: Resolution of maps (meters).
        """

        self.data_path = Path(data_path).expanduser().absolute()
        self.json_path = Path(json_path)

        self.table_names = [
            "category",
            "attribute",
            "sensor",
            "calibrated_sensor",
            "ego_pose",
            "log",
            "scene",
            "sample",
            "sample_data",
            "map",
        ]

        start_time = time.time()

        # Explicitly assign tables to help the IDE determine valid class members.
        self.category = self.__load_table__("category")
        self.attribute = self.__load_table__("attribute")

        self.sensor = self.__load_table__("sensor")
        self.calibrated_sensor = self.__load_table__("calibrated_sensor")
        self.ego_pose = self.__load_table__("ego_pose")
        self.log = self.__load_table__("log")
        self.scene = self.__load_table__("scene")
        self.sample = self.__load_table__("sample")
        self.sample_data = self.__load_table__("sample_data")

        self.map = self.__load_table__("map")

        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.1f} seconds.\n======".format(time.time() - start_time))

        # Initialize LyftDatasetExplorer class
        self.explorer = LyftDatasetExplorer(self)
        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

    def __make_reverse_index__(self, verbose: bool) -> None:
        """De-normalizes database to create reverse indices for common cases.
        Args:
            verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member["token"]] = ind

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get("calibrated_sensor", record["calibrated_sensor_token"])
            sensor_record = self.get("sensor", cs_record["sensor_token"])
            record["sensor_modality"] = sensor_record["modality"]
            record["channel"] = sensor_record["channel"]

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record["data"] = {}
            record["anns"] = []

        for record in self.sample_data:
            if record["is_key_frame"]:
                sample_record = self.get("sample", record["sample_token"])
                sample_record["data"][record["channel"]] = record["token"]

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))


class_heights = {'animal': 0.51, 'bicycle': 1.44, 'bus': 3.44, 'car': 1.72, 'emergency_vehicle': 2.39,
                 'motorcycle': 1.59,
                 'other_vehicle': 3.23, 'pedestrian': 1.78, 'truck': 3.44}


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
        corners_voxel = car_to_voxel_coords(corners, im.shape, voxel_size, z_offset).transpose(1, 0)
        corners_voxel = corners_voxel[:, :2]  # Drop z coord

        class_color = classes.index(box.name) + 1

        if class_color == 0:
            raise Exception("Unknown class: {}".format(box.name))

        cv2.drawContours(im, np.int0([corners_voxel]), 0, (class_color, class_color, class_color), -1)

# Some hyperparameters we'll need to define for the system

# We scale down each box so they are more separated when projected into our coarse voxel space.
box_scale = 0.8


def car_to_voxel_coords(points, shape, voxel_size, z_offset=0):
    if len(shape) != 3:
        raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")

    if len(points.shape) != 2 or points.shape[0] not in [3, 4]:
        raise Exception("Input points should be (3,N) or (4,N) in shape, found {}".format(points.shape))

    tm = create_transformation_matrix_to_voxel_space(shape, voxel_size, (0, 0, z_offset))
    p = transform_points(points, tm)
    return p


def create_voxel_pointcloud(points, shape, voxel_size=(0.5, 0.5, 1), z_offset=0):
    points_voxel_coords = car_to_voxel_coords(points.copy(), shape, voxel_size, z_offset)
    points_voxel_coords = points_voxel_coords[:3].transpose(1, 0)
    points_voxel_coords = np.int0(points_voxel_coords)

    bev = np.zeros(shape, dtype=np.float32)
    bev_shape = np.array(shape)

    within_bounds = (np.all(points_voxel_coords >= 0, axis=1) * np.all(points_voxel_coords < bev_shape, axis=1))

    points_voxel_coords = points_voxel_coords[within_bounds]
    coord, count = np.unique(points_voxel_coords, axis=0, return_counts=True)

    # Note X and Y are flipped:
    bev[coord[:, 1], coord[:, 0], coord[:, 2]] = count

    return bev


def normalize_voxel_intensities(bev, max_intensity=16):
    return (bev / max_intensity).clip(0, 1)


def create_transformation_matrix_to_voxel_space(shape, voxel_size, offset):
    """
    Constructs a transformation matrix given an output voxel shape such that (0,0,0) ends up in the center.
    Voxel_size defines how large every voxel is in world coordinate, (1,1,1) would be the same as Minecraft voxels.
    An offset per axis in world coordinates (metric) can be provided, this is useful for Z (up-down) in lidar points.
    """

    shape, voxel_size, offset = np.array(shape), np.array(voxel_size), np.array(offset)

    tm = np.eye(4, dtype=np.float32)
    translation = shape / 2 + offset / voxel_size

    tm = tm * np.array(np.hstack((1 / voxel_size, [1])))
    tm[:3, 3] = np.transpose(translation)
    return tm


def transform_points(points, transf_matrix):
    """
    Transform (3,N) or (4,N) points using transformation matrix.
    """
    if points.shape[0] not in [3, 4]:
        raise Exception("Points input should be (3,N) or (4,N) shape, received {}".format(points.shape))
    return transf_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]


# Let's try it with some example values
tm = create_transformation_matrix_to_voxel_space(shape=(100, 100, 4), voxel_size=(0.5, 0.5, 0.5), offset=(0, 0, 0.5))
p = transform_points(np.array([[10, 10, 0, 0, 0], [10, 5, 0, 0, 0], [0, 0, 0, 2, 0]], dtype=np.float32), tm)
print(p)

def visualize_lidar_of_sample(sample_token, axes_limit=80):
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    level5data.render_sample_data(sample_lidar_token, axes_limit=axes_limit)

class BEVImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_filepaths, transforms=None):
        self.input_filepaths = input_filepaths
        self.transforms = transforms

    def __len__(self):
        return len(self.input_filepaths)

    def __getitem__(self, idx):
        input_filepath = self.input_filepaths[idx]

        sample_token = input_filepath.split("/")[-1].replace("_input.png", "")

        im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)

        if self.transforms is not None:
            augmented = self.transforms(image=im)
            im = augmented['image']

        im = im.astype(np.float32) / 255

        im = torch.from_numpy(im.transpose(2, 0, 1))

        return im, sample_token

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

df.head()
NUM_WORKERS = os.cpu_count() * 3

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
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))
            sample_token = sample["next"]
            continue

        bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
        bev = normalize_voxel_intensities(bev)

        boxes = level5data.get_boxes(sample_lidar_token)

        target = np.zeros_like(bev)

        move_boxes_to_car_space(boxes, ego_pose)
        scale_boxes(boxes, box_scale)
        draw_boxes(target, voxel_size, boxes=boxes, classes=classes, z_offset=z_offset)

        bev_im = np.round(bev * 255).astype(np.uint8)
        target_im = target[:, :, 0]  # take one channel only

        cv2.imwrite(os.path.join(output_folder, "{}_input.png".format(sample_token)), bev_im)
        cv2.imwrite(os.path.join(output_folder, "{}_target.png".format(sample_token)), target_im)

        sample_token = sample["next"]


# for df, data_folder in [(df, test_data_folder)]:
#     print("Preparing data into {} using {} workers".format(data_folder, NUM_WORKERS))
#     first_samples = df.first_sample_token.values
#     os.makedirs(data_folder, exist_ok=True)
#     process_func = partial(prepare_training_data_for_scene,
#                            output_folder=data_folder, bev_shape=bev_shape, voxel_size=voxel_size, z_offset=z_offset, box_scale=box_scale)
#     pool = Pool(NUM_WORKERS)
#     for _ in tqdm_notebook(pool.imap_unordered(process_func, first_samples), total=len(first_samples)):
#         pass
#     pool.close()
#     del pool


# %%

input_filepaths = sorted(glob.glob(os.path.join(test_data_folder, "*_input.png")))
import albumentations as A

albu_test_tansforms = A.Compose([  # no augmentations
    A.Normalize(),
])
test_dataset = BEVImageDataset(input_filepaths, transforms=albu_test_tansforms)

im, sample_token = test_dataset[1]
im = im.numpy()

# Transpose the input volume CXY to XYC order, which is what matplotlib requires.
visualize_lidar_of_sample(sample_token)

def visualize_predictions(input_image, prediction, n_images=2, apply_softmax=True):
    """
    Takes as input 3 PyTorch tensors, plots the input image, predictions and targets.
    """
    # Only select the first n images
    prediction = prediction[:n_images]

    input_image = input_image[:n_images]

    prediction = prediction.detach().cpu().numpy()
    if apply_softmax:
        prediction = scipy.special.softmax(prediction, axis=1)
    class_one_preds = np.hstack(1 - prediction[:, 0])

    class_rgb = np.repeat(class_one_preds[..., None], 3, axis=2)
    class_rgb[..., 2] = 0

    input_im = np.hstack(input_image.cpu().numpy().transpose(0, 2, 3, 1))

    if input_im.shape[2] == 3:
        input_im_grayscale = np.repeat(input_im.mean(axis=2)[..., None], 3, axis=2)
        overlayed_im = (input_im_grayscale * 0.6 + class_rgb * 0.7).clip(0, 1)
    else:
        input_map = input_im[..., 3:]
        overlayed_im = (input_map * 0.6 + class_rgb * 0.7).clip(0, 1)

    thresholded_pred = np.repeat(class_one_preds[..., None] > 0.5, 3, axis=2)

def calc_detection_box(prediction_opened, class_probability):
    sample_boxes = []
    sample_detection_scores = []
    sample_detection_classes = []

    contours, hierarchy = cv2.findContours(prediction_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)

        # Let's take the center pixel value as the confidence value
        box_center_index = np.int0(np.mean(box, axis=0))

        for class_index in range(len(classes)):
            box_center_value = class_probability[class_index + 1, box_center_index[1], box_center_index[0]]

            # Let's remove candidates with very low probability
            if box_center_value < 0.01:
                continue

            box_center_class = classes[class_index]

            box_detection_score = box_center_value
            sample_detection_classes.append(box_center_class)
            sample_detection_scores.append(box_detection_score)
            sample_boxes.append(box)

    return np.array(sample_boxes), sample_detection_scores, sample_detection_classes

# We perform an opening morphological operation to filter tiny detections
# Note that this may be problematic for classes that are inherently small (e.g. pedestrians)..
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def open_preds(predictions_non_class0):
    predictions_opened = np.zeros((predictions_non_class0.shape), dtype=np.uint8)

    for i, p in enumerate(tqdm(predictions_non_class0)):
        thresholded_p = (p > background_threshold).astype(np.uint8)
        predictions_opened[i] = cv2.morphologyEx(thresholded_p, cv2.MORPH_OPEN, kernel)

    return predictions_opened


# Test Set Predictions


import gc

gc.collect()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=os.cpu_count() * 2)
progress_bar = tqdm_notebook(test_loader)
# progress_bar = tqdm(test_loader)

# We quantize to uint8 here to conserve memory. We're allocating >20GB of memory otherwise.
# predictions = np.zeros((len(test_loader), 1+len(classes), 336, 336), dtype=np.uint8)

sample_tokens = []
all_losses = []

detection_boxes = []
detection_scores = []
detection_classes = []

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

#         # Visualize the first prediction
#         if ii == 0:
#             visualize_predictions(X, prediction, apply_softmax=False)


print("Total amount of boxes:", np.sum([len(x) for x in detection_boxes]))

ind = 11
# Visualize the boxes in the first sample
t = np.zeros_like(predictions_opened[0])
for sample_boxes in detection_boxes[ind]:
    box_pix = np.int0(sample_boxes)
    cv2.drawContours(t, [box_pix], 0, (255), 2)

def create_transformation_matrix_to_voxel_space(shape, voxel_size, offset):
    """
    Constructs a transformation matrix given an output voxel shape such that (0,0,0) ends up in the center.
    Voxel_size defines how large every voxel is in world coordinate, (1,1,1) would be the same as Minecraft voxels.
    An offset per axis in world coordinates (metric) can be provided, this is useful for Z (up-down) in lidar points.
    """

    shape, voxel_size, offset = np.array(shape), np.array(voxel_size), np.array(offset)

    tm = np.eye(4, dtype=np.float32)
    translation = shape / 2 + offset / voxel_size

    tm = tm * np.array(np.hstack((1 / voxel_size, [1])))
    tm[:3, 3] = np.transpose(translation)
    return tm


def transform_points(points, transf_matrix):
    """
    Transform (3,N) or (4,N) points using transformation matrix.
    """
    if points.shape[0] not in [3, 4]:
        raise Exception("Points input should be (3,N) or (4,N) shape, received {}".format(points.shape))
    return transf_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]


def car_to_voxel_coords(points, shape, voxel_size, z_offset=0):
    if len(shape) != 3:
        raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")

    if len(points.shape) != 2 or points.shape[0] not in [3, 4]:
        raise Exception("Input points should be (3,N) or (4,N) in shape, found {}".format(points.shape))

    tm = create_transformation_matrix_to_voxel_space(shape, voxel_size, (0, 0, z_offset))
    p = transform_points(points, tm)
    return p


def create_voxel_pointcloud(points, shape, voxel_size=(0.5, 0.5, 1), z_offset=0):
    points_voxel_coords = car_to_voxel_coords(points.copy(), shape, voxel_size, z_offset)
    points_voxel_coords = points_voxel_coords[:3].transpose(1, 0)
    points_voxel_coords = np.int0(points_voxel_coords)

    bev = np.zeros(shape, dtype=np.float32)
    bev_shape = np.array(shape)

    within_bounds = (np.all(points_voxel_coords >= 0, axis=1) * np.all(points_voxel_coords < bev_shape, axis=1))

    points_voxel_coords = points_voxel_coords[within_bounds]
    coord, count = np.unique(points_voxel_coords, axis=0, return_counts=True)

    # Note X and Y are flipped:
    bev[coord[:, 1], coord[:, 0], coord[:, 2]] = count

    return bev


def normalize_voxel_intensities(bev, max_intensity=16):
    return (bev / max_intensity).clip(0, 1)


from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D

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

    #     box_height = 1.75
    box_height = np.array([class_heights[cls] for cls in sample_detection_class])

    # Note: Each of these boxes describes the ground corners of a 3D box.
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
sub.to_csv('lyft3d_pred.csv', index=False)