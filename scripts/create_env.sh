#!/bin/bash

conda create -y -n lyft python=3.6
source activate lyft

conda install -y -n rsna pytorch=0.4.1 cuda90 -c pytorch
pip install --upgrade pip
pip install -r requirements.txt
