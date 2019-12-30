import os
import random
import sys

import albumentations as A
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import label2rgb
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomCrop, RandomHorizontalFlip,
    RandomVerticalFlip, Resize, ToTensor)

from configs import BEV_SHAPE, DATA_ROOT, IMG_SIZE, NUM_CLASSES, OUTPUT_ROOT, PROJECT_ROOT

sys.path.append(PROJECT_ROOT)



BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

train_transform = Compose([
    RandomCrop(IMG_SIZE),
    RandomHorizontalFlip(0.5), 
    RandomVerticalFlip(0.5),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensor()
    ])


test_transform = Compose([
    CenterCrop(IMG_SIZE),
    RandomHorizontalFlip(0.5),
    RandomVerticalFlip(0.5),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensor()
    ])  


tensor_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


crop_d4_transforms = A.Compose([
			            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, 
                        interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                    	# D4 Group augmentations
                    	A.HorizontalFlip(p=0.5),
                    	A.VerticalFlip(p=0.5),
                    	A.RandomRotate90(p=0.5),
                    	A.Transpose(p=0.2),
                    	# crop and resize  
                    	A.RandomSizedCrop((BEV_SHAPE[0]-50, BEV_SHAPE[0]), IMG_SIZE, IMG_SIZE, w2h_ratio=1.0, 
                                        interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.2),                       
                    	A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR, p=1),      
                    	A.Normalize(),
                    	])


albu_show_transforms  = A.Compose([
			            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, 
                        interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                    	# D4 Group augmentations
                    	A.HorizontalFlip(p=0.5),
                    	A.VerticalFlip(p=0.5),
                    	A.RandomRotate90(p=0.5),
                    	A.Transpose(p=0.2),
                    	# crop and resize  
                    	A.RandomSizedCrop((BEV_SHAPE[0]-50, BEV_SHAPE[0]), IMG_SIZE, IMG_SIZE, w2h_ratio=1.0, 
                                        interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.2),                       
                    	A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR, p=1),      
                    	])

albu_valid_tansforms = A.Compose([# D4 Group augmentations
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.Transpose(p=0.5),        
                        A.Normalize(),
			    	    ])


albu_test_tansforms = A.Compose([# no augmentations                            
                        A.Normalize(),
			    	    ])            

   
D4_transforms = [# D4 Group augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),        
                A.Normalize()
                ]

train_transforms = [A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, 
                       interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                    # D4 Group augmentations
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.2),
                    # crop and resize  
		    #A.RandomSizedCrop((BEV_SHAPE[0]-100, BEV_SHAPE), IMG_SIZE, IMG_SIZE, w2h_ratio=1.0, 
                    #                    interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.2),                       
                    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR, p=1),      
                    A.Normalize(),
                    ]


valid_transforms = [# D4 Group augmentations
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.2),                    
                    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1),
                    A.Normalize(),
                    ]

light_augs = A.Compose([
        # D4 Group augmentations
        A.HorizontalFlip(p=0.5),
	    A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),       
        # ligth augs
        A.RandomScale(scale_limit=0.2), 
        A.Rotate(),
        A.RandomSizedCrop((IMG_SIZE - 100, IMG_SIZE), IMG_SIZE, IMG_SIZE),
        A.Normalize()
    ], bbox_params={'format':'coco', 'min_area': 1, 'min_visibility': 0.5, 'label_fields': ['category_id']}, p=1)


def visualize_bbox(img, bbox, color=(255, 255, 0), thickness=2):  
    """Helper to add bboxes to images 
    Args:
        img : image as open-cv numpy array
        bbox : boxes as a list or numpy array in pascal_voc fromat [x_min, y_min, x_max, y_max]  
        color=(255, 255, 0): boxes color 
        thickness=2 : boxes line thickness
    """
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img

def augment_and_show(aug, image, masks, bboxes=[], categories=[], category_id_to_name=[], filename=None, 
                     font_scale_orig=0.35, font_scale_aug=0.35, 
                     show_title=True, **kwargs):

    augmented = aug(image=image, masks=masks, bboxes=bboxes, category_id=categories)

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_aug = augmented['image'] #cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)

    for bbox in bboxes:
        visualize_bbox(image, bbox)

    for bbox in augmented['bboxes']:
        visualize_bbox(image_aug, bbox)

    one_mask = np.zeros_like(masks[0])
    for i, mask in enumerate(masks):
        one_mask += (mask > 0).astype(np.uint8) * i

    aug_masks = augmented['masks']
    one_aug_mask = np.zeros_like(aug_masks[0])
    for i, augmask in enumerate(aug_masks):
        one_aug_mask += (augmask > 0).astype(np.uint8) * i 

    # extend mask for three channels        
    if len(one_aug_mask.shape) != 3:
        mask = label2rgb(one_mask, bg_label=0)            
        mask_aug = label2rgb(one_aug_mask, bg_label=0)  

    f, ax = plt.subplots(2, 2, figsize=(16, 16))       
    ax[0, 0].imshow(image)
    ax[0, 0].set_title('Original image')        
    ax[0, 1].imshow(image_aug)
    ax[0, 1].set_title('Augmented image')        
    ax[1, 0].imshow(mask, interpolation='nearest')
    ax[1, 0].set_title('Original mask')
    ax[1, 1].imshow(mask_aug, interpolation='nearest')
    ax[1, 1].set_title('Augmented mask')
    f.tight_layout()
    plt.show()

    if filename is not None:
        f.savefig(filename)
        
    return augmented['image'], augmented['masks'], augmented['bboxes']

def find_in_dir(dirname):
    return [os.path.join(dirname, fname) for fname in sorted(os.listdir(dirname))]
