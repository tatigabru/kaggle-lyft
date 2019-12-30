from datetime import datetime
from functools import partial
import glob
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm, tqdm_notebook
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R
import os

def plot_img_target_map(bev, map_img, num = 0):
    """Helper to vizualize img, target and map"""
    plt.figure(num, figsize=(16,8))
    plt.imshow(np.hstack((bev, map_img))) 
    


def main():

    data_folder = 'C:/Users/New/Documents/Challenges/lyft/input/maps/bev_train_data_336'
    my_data_folder = 'C:/Users/New/Documents/Challenges/output/bev_data'
    # choose inputs/targets
    input_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_input.png")))
    sample_tokens = [x.split("/")[-1].replace("_input.png","") for x in input_filepaths] 
    sample_tokens = [x.split("bev_train_data_336\\")[-1].replace("_input.png","") for x in input_filepaths]   
    print(sample_tokens[:5])
    num = 0
    sample_token = sample_tokens[0]

    input_filepath = '{}/{}_input.png'.format(data_folder, sample_token)
    map_filepath = '{}/{}_map.png'.format(data_folder, sample_token)
    target_filepath = '{}/{}_target.png'.format(data_folder, sample_token)
    
    img336 = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)             
    map_img = cv2.imread(map_filepath, cv2.IMREAD_UNCHANGED)
    target336 = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)

    # plot 336 resolution
    plt.figure(num, figsize=(16,8))
    plt.imshow(np.hstack((img336, map_img))) 
    plt.show()

    input_filepath = '{}/{}_input.png'.format(my_data_folder, sample_token)
    target_filepath = '{}/{}_target.png'.format(my_data_folder, sample_token)

    img768 = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED) 
    target768 = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)

    padding = int((768-672)//2)         
    # resize and test
    img_new= cv2.resize(img336, (336*2, 336*2))    
    target_new= cv2.resize(target336, (336*2, 336*2))  
    img_new = cv2.copyMakeBorder(img_new, padding , padding, padding, padding, cv2.BORDER_CONSTANT,value=0) 
    target_new = cv2.copyMakeBorder(target_new, padding , padding, padding, padding, cv2.BORDER_CONSTANT,value=0) 


    plt.figure(3, figsize=(16,8))
    plt.imshow(np.hstack((img_new, img768))) 
    plt.show()
 
    plt.figure(4, figsize=(16,8))
    plt.imshow(np.hstack((target_new, target768))) 
    plt.show()


if __name__ == '__main__':
    main()    