# -*- coding: utf-8 -*-
"""
Created on Wed May 22 00:56:32 2019

inference of mask-rcnn

"""

import os
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
import numpy as np

IMG_DIR = '/home/tanya/coco/train2014/'
files = os.listdir(IMG_DIR)
print(files[0])

config_file = "/home/tanya/ifashion/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
#cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

# load image as npy array and then run prediction
image = cv2.imread(IMG_DIR+files[0])
print(image.shape)

predictions = coco_demo.run_on_opencv_image(image)
print(predictions)