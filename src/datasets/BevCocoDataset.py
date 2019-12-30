import collections
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFile
import pandas as pd
from tqdm import tqdm
from numba import jit

from transforms import (train_transform, test_transform, tensor_transform)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BevCocoDataset(torch.utils.data.Dataset):
    """
    Bird-eye-view dataset amended for coco style
    
        :param fold: integer, number of the fold
        :param df: Dataframe with sample tokens
        :param debug: if True, runs the debugging on few images
        :param img_size: the desired image size to resize to        
        :param input_dir: directory with imputs and targets (and maps, optionally)
        :param if_map: if True, maps are added   
        """    
    def __init__(self, fold: int, df: pd.DataFrame, 
                 debug: bool, img_size: int, 
                 input_dir: str, transforms = None, if_map = False):
        super(BEVImageDataset, self).__init__()  # inherit it from torch Dataset
        self.fold = fold
        self.df = df
        self.debug = debug
        self.img_size = img_size
        self.input_dir = input_dir
        self.transforms = transforms
        self.if_map = if_map

        if self.debug:
            self.df = self.df.head(16)
            print('Debug mode, samples: ', self.df.samples)  

        self.sample_tokens = list(self.df.samples)

    def __len__(self):
        return len(self.sample_tokens)
    
    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]
        input_filepath = '{}/{}_input.png'.format(self.input_dir, sample_token)
        
        # load PIL image
        img = Image.open(img_path).convert("RGB")
        #img = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED) 
        #img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 

        # load targets 
        target_filepath = '{}/{}_coco_target.pickle'.format(self.input_dir, sample_token)    
        with open(target_filepath, 'rb') as handle:
            targets = pickle.load(handle)    
       
        # augment
        if self.transforms is not None:
            img, targets = self.transforms(img, targets)

        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)            
        masks = targets["masks"]
        boxes = targets["boxes"]
        labels = targets["labels"]           

        # remove tiny instances
        num_objs = len(labels)
        new_boxes = []
        new_labels = []
        new_masks = []        
        for num in range(num_objs):
            [xmin, ymin, xmax, ymax] = boxes[num, :]
            if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                new_boxes.append([xmin, ymin, xmax, ymax])
                new_labels.append(labels[num])
                new_masks.append(masks[num, :, :])         
        if len(new_labels) == 0:
            print('no instances left')
            new_boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(masks[0, :, :])        
        nmx = np.zeros((len(new_masks), masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for i, mask in enumerate(new_masks): 
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)         
            nmx[i, :, :] = mask

        # to tensor
        boxes = torch.as_tensor(new_boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd       

        return img, target


def test_dataset_augs(train_df, data_folder):
    # dataset
    train_dataset = BevCocoDataset(fold=3, df=train_df, 
                                    debug=True, img_size=IMG_SIZE, 
                                    input_dir=data_folder, 
                                    transforms = train_transforms)
    for count in range(10):
        # get dataset sample and plot it
        img, target = train_dataset[0]  
        w,h = img.size      
        masks = target["masks"].numpy()  
        mask = masks[0, :, ;]*255
        target_as_rgb = np.repeat(mask[...,None], 3, 2) # repeat array for three channels
        img = np.array(img).reshape((w,h))

        plt.figure(count+1, figsize=(16,8))    
        # Transpose the input volume CXY to XYC order, which is what matplotlib requires.
        plt.imshow(np.hstack((img.transpose(1,2,0)[...,:3], target_as_rgb))) 
        plt.title(sample_token)
        plt.show()


def main():

    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    num_cls = len(classes)

    # coco folder
    data_folder = os.path.join(OUTPUT_ROOT, "coco_data_768")
    input_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_input.png")))
    sample_tokens = [x.split("/")[-1].replace("_input.png","") for x in input_filepaths]
    sample_tokens = [x.replace("bev_data\\","") for x in sample_tokens]
    
    # train samples
    df = pd.read_csv('train_samples.csv')
    print(df.head())

    train_df = df[df['samples'].isin(sample_tokens)]
    print(train_df.head())
