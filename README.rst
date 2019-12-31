Lyft 3D Object Detection for Autonomous Vehicles
=====================

https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/overview

.. contents::

Install
=======
Use Python 3.6.

Install:
    install anaconda

    create environment (use scripts/bash create_env.sh)
or
    pip install -r requirements.txt
or 
    use Docker    

Overview
========
General approach is the following:

- Dataset is split into 4 folds by scene stratified by cars
- Lidar point coulds are voxelized and then projected into the bird-eye-view (BEV) along with bounding boxes 
- BEV of images and bboxes are used to generade 2D input images and corresponding masks at various resolutions: 768x768, 1024x1024 and 2048x2048 
- Unet-like architectures with different encoders where used for segmentation. I tried ``resnet152``, ``resnet101``, ``se-resnext101``, ``resnet50``, and other backbones pretrained on ImageNet.
- The predicted masks were post-processed with OpenCV libruary to obtain rotated bonding boxes
- Simple heuristics were used to translate 2D bounding boxes into 3D ones, using ground level, meta data and that fact, that boxes are mostly vertical
- Mask-RCNN was considered as an alternative for Unet type models (it was too slow and performed worse) 
- A "classification" model head was added to the Unet-type architechture (did not had enoupg time to experiment with it more)
- Maps could be also added on top of the image to improve the accuracy


Dataset
------------
The task required quite a pre-processing. The src/preprocessing containes scripts for BEV images  preparation from the Lidar point clouds for both train and test. 

For simplicity, I uploaded generated BEV images and corresponding masks here:
 
https://www.kaggle.com/blondinka/bev-train-test

Folds
------------
The folds are in src/folds/ directory. File src/make_folds.py contains code with several examples of splits with various stratification strategies. I used 4 folds split by scenes and stratified by cars.


Augmentations
------------
Augmentations were implemented with the help of albumentations libruary. 
I used
* D4 augmentations: horizontal/vertical flip, 90% rotate and transpose
* Rescale and crops
* Then resizeand noramisation

The lists of used augmentations for train and validation are in src/datasets/transforms.py

Training
--------------

The main unet training script is ``src/train.py``. I used a classical unet model as a baseline and then experiemnted with Unet with different backbones from segmentation_models_pytoch libruary by qubvel.

I also tried to use Mask-RCNN from torchvision, the train runner is in ``src/train_mask.py``. However, I did not perform that well and was considerably slower.

I tried a two-head model with both classification and segmentation. The runner for it is ``src/train_seg_class.py``

I used progressive learning 512-768-1024 to speed up the training. 
Radam optimiser and multi-step learning rate scheduler.

Inference
----------------
Test BEV images can be downloaded here:

https://www.kaggle.com/blondinka/bev-train-test

The inference is in ``src/inference.py`` file 

Some details:
* Progressive learning helpedto improve the accuracy of the model
* Surprisingly, heavy backbones, i.e. ``resnet152`` did not perform that well for this task, same for the Mask-RCNN model from torchvision
* Unet type achitectures were better than FPN
* Augmentations helped, as always
* Better resolution led to better results (as expected)

References:
----------------
Lyft SDK: https://github.com/lyft/nuscenes-devkit/tree/master/notebooks

https://www.kaggle.com/gzuidhof/reference-model

Unet: https://arxiv.org/abs/1505.04597

Quaternions: http://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf


