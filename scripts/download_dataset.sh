#!/usr/bin/env bash

pip install --upgrade pip
CUR_DIR=$pwd
DATA_DIR_LOC=dataset

cd ..
mkdir -p $DATA_DIR_LOC
cd $DATA_DIR_LOC

if [ "$(ls -A $(pwd))" ]
then
    echo "$(pwd) not empty!"
else
    echo "$(pwd) is empty!"
    pip install kaggle --upgrade
    kaggle competitions download -c 3d-object-detection-for-autonomous-vehicles
    mkdir train_data
    mkdir test_data
    unzip train_data.zip -d train_data
    unzip test_data.zip -d test_data
    mkdir train_images
    mkdir test_images
    unzip train_images.zip -d train_images
    mkdir train_lidar
    mkdir test_lidar
    unzip train_lidar.zip -d train_lidar
    unzip test_lidar.zip -d test_lidar
    mkdir train_map
    mkdir test_map
    unzip train_lidar.zip -d train_map
    unzip test_lidar.zip -d test_map   
fi

cd $CUR_DIR
echo $(pwd)
