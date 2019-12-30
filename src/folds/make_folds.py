"""
Make folds

"""
import argparse
import copy
import json
import math
import os.path
import sys
from pathlib import Path
sys.path.append('/home/user/challenges/lyft/lyft_repo/src')

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from configs import DATA_ROOT
from pyquaternion import Quaternion


class Table:
    def __init__(self, data):
        self.data = data
        self.index = {x['token']: x for x in data}

def load_table(name, root=os.path.join(DATA_ROOT, 'train_data')):
    with open(os.path.join(root, name), 'rb') as f:
        return Table(json.load(f))


def get_scene_samples(scene: Table, sample: Table, if_save = False):
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


def make_split_by_car(df: pd.DataFrame):
    """
    Make train - validation split by cars (hosts) 

    Args: 
        df = pd.DataFrame(columns=["host", "scene_name", "date", 
                                "scene_token", "first_sample_token"])    
    """
    hosts = df["host"].unique()
    print(hosts)
    
    # split cars (hosts)
    train_hosts, validation_hosts = train_test_split(
                                hosts, test_size=0.25, random_state=413)
    print('train hosts {}, val hosts {}'.format(len(train_hosts), len(validation_hosts)))

    validation_df = df[df["host"].isin(validation_hosts)]
    vi = validation_df.index
    train_df = df[~df.index.isin(vi)]
    print(len(train_df), len(validation_df), "train/validation split scene counts")

    return train_df, validation_df


def make_split_by_scene(df: pd.DataFrame):
    """ 
    Make train - validation split by scenes

    Args: 
        df = pd.DataFrame(columns=["host", "scene_name", "date", 
                                "scene_token", "first_sample_token"])    
    """
    scene_tokens = df["scene_token"].values
    #print(scene_tokens)

    # split scenes
    scenes_train, scenes_val = train_test_split(
                                scene_tokens, test_size=0.25, random_state=413)
    print('train scenes {}, val scenes {}'.format(len(scenes_train), len(scenes_val)))

    validation_df = df[df["scene_token"].isin(scenes_val)]
    vi = validation_df.index
    train_df = df[~df.index.isin(vi)]
    print(len(train_df), len(validation_df), "train/validation split scene counts")
    
    # sanity checks
    host_count_df = validation_df.groupby("host")['scene_token'].count()
    print('validation_df host counts {}'.format(host_count_df))

    return train_df, validation_df


def split_by_scene_stratify_hosts(df: pd.DataFrame, if_save = False):
    """ 
    Make train - validation split by scenes, stratified by host

    Args: 
        df = pd.DataFrame(columns=["host", "scene_name", "date", 
                                "scene_token", "first_sample_token"]) 
        if_save: boolean flag weather to save the folds dataframes                           
    """
    df['folds'] = 0
    scene_tokens = df["scene_token"].values
    hosts = df["host"].values
    skf = StratifiedKFold(n_splits=4, random_state=413, shuffle=True)

    # split scenes stratified by car
    for num, (train_index, test_index) in enumerate(skf.split(scene_tokens, hosts)):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_val = scene_tokens[train_index], scene_tokens[test_index]
        y_train, y_test = hosts[train_index], hosts[test_index] 
        # save folds to scv file    
        df['folds'].iloc[test_index] = num

    print(df.head(10))
    print(df['folds'].unique())   

    print('train scenes {}, val scenes {}'.format(len(X_train), len(X_val)))
    validation_df = df[df["scene_token"].isin(X_val)]
    vi = validation_df.index
    train_df = df[~df.index.isin(vi)]
        
    # sanity checks
    host_count_df = validation_df.groupby("host")['scene_token'].count()
    print('validation_df host counts {}'.format(host_count_df))
    host_count_df = train_df.groupby("host")['scene_token'].count()
    print('train_df host counts {}'.format(host_count_df))

    if if_save:
        df.to_csv('scenes_folds.csv', index = False)

    return df, train_df, validation_df


def main():
    # get scenes and hosts
    df = pd.read_csv('host_scenes.csv')
    host_count_df = df.groupby("host")['scene_token'].count()
    print(host_count_df)

    # make folds
   # train_df, validation_df = make_split_by_car(df)
   # train_df, validation_df = make_split_by_scene(df)       
   # df, train_df, validation_df = split_by_scene_stratify_hosts(df, True)

    # scenes tokens for train and validation
    df = pd.read_csv('scenes_folds.csv')    
       
    # load scenes and samples
    scene = load_table('scene.json')
    sample = load_table('sample.json') 
    
    # get all sample tokens for all scenes
    all_tokens = get_scene_samples(scene, sample, if_save = False)  

    for fold in range(3):
        train_df = df[df['folds'] != fold]
        validation_df = df[df['folds'] == fold]
        scenes_train = train_df["scene_token"].values
        scenes_val = validation_df["scene_token"].values
        print('train scenes {}, val scenes {}'.format(len(scenes_train), len(scenes_val)))

        # get samples tokens for train and val scenes
        train_tokens = [all_tokens[scenes_train[x]] for x in range(len(scenes_train))]
        train_tokens = np.array(train_tokens)
        train_tokens = np.concatenate(train_tokens)

        val_tokens = [all_tokens[scenes_val[x]] for x in range(len(scenes_val))]
        val_tokens = np.array(val_tokens)
        val_tokens = np.concatenate(val_tokens)
        print('train_tokens.shape {}, val_tokens.shape {}'.format(train_tokens.shape, val_tokens.shape))

        # save train and validation sample tokens
        train_samples = pd.DataFrame()
        train_samples['samples'] = train_tokens
        train_samples.to_csv(f'train_fold_{fold}.csv', index = False)

        val_samples = pd.DataFrame()
        val_samples['samples'] = val_tokens
        val_samples.to_csv(f'val_fold_{fold}.csv', index = False)   


if __name__ == '__main__':   
    main()

    