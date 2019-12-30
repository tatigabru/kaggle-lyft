ON_KAGGLE = False
ON_SERVER = True

DATA_ROOT = '/kaggle/input/3d-object-detection-for-autonomous-vehicles/' if ON_KAGGLE else '../input/'
DATA_DIR = '../../input/'
OUTPUT_ROOT = "../../../output/"
PROJECT_ROOT = "/home/tanya/lyft/progs/kaggle-lyft" if ON_SERVER else 'C:/Users/New/Documents/Challenges/lyft/progs'

IMG_SIZE = 512
BEV_SHAPE = (768, 768, 3)
VOXEL_SIZE = (0.2, 0.2, 1.5)
Z_OFFSET = -2.0

CLASSES = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]      
NUM_CLASSES = 9

