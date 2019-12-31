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
- Unet-like architectures with different encoders where used for segmentation. I tried ``resnet152``, ``resnet101``, ``se-resnext101``, ``resnet50``, ``se-resnext50``and some other backbones pretrained on ImageNet.
- The predicted masks were post-processed with OpenCV libruary to obtain rotated bonding boxes
- Simple heuristics were used to translate 2D bounding boxes into 3D ones, using ground level, meta data and that fact, that boxes are mostly vertical
- Mask-RCNN was considered as an alternative for Unet type models (it was too slow and performed worse) 
- A "classification" model head was added to the Unet-type architechture (did not had enoupg time to experiment with it more)
- Maps could be also added on top of the image to improvethe accuracy


Dataset
------------
The task required quite a pre-processing. The src/preprocessing containes scripts for BEV images  preparation from the Lidar point clouds for both train and test. 

For simplicity, I uploaded generated BEV images and corresponding masks here: 
https://www.kaggle.com/blondinka/bev-train-test

Folds
------------
The folds are in src/folds/ directory. File src/make_folds.py contains code with several examples of splits with various stratification strategies. I used 4 folds split by scene and stratified by car.



* augmentations used: scale, minor color augmentations
  (hue/saturation/value), Albumentations library was used.

 

Segmentation into characters is done with a Faster-RCNN model with ``resnet152``
backbone trained with torchvision. Only one class is used, so it does not
try to predict the character class. This model trains very fast and gives
high quality boxes. Competition F1 metric (assuming
perfect prediction for the classes) was around ~0.99 on validation.

Some details:

* torchvision detection pipeline was adapted,
* ``resnet152`` backbone worked a bit better than default ``resnet50`` (even though
  it was not pre-trained on COCO, doing this would offer another small boost),
* pipeline was modified to accept empty crops (crops without ground truth
  objects) to reduce amount of false positives,
* it was trained on 512x384 crops, with page height around 1500 px,
  and full pages were used for inference,


Overall many more improvements are possible here: using mmdetection,
better models, pre-training on COCO, blending predictions from different folds
for submission, TTA, separate model to discard out-of-page symbols, etc.
Still it seemed that classification was more important.

See ``kuzushiji.segment``, which is based on torchvision detection reference,
and ``kuzushiji.segment.dataset`` for the dataset.

Classification
--------------

Classification is performed by a model which gets as input a large crop
from the image (512x768 from 2500-3000px high image) which contains multiple
characters. It also recieves as input boudning boxes predicted by segmentation
model (these are out-of-fold predictions). This is similar to multi-class
detection, but with frozen bounding boxes.
ResNet base is used, ``layer4`` is discarded, features are extracted for each
bounding box with ``roi_align`` from ``layer2`` and ``layer3`` and concatenated,
and then passed into a classification head.

Some details:

* surprisingly, details such as architecture, backbone and learning regime
  made a lot of difference, much more than usual.
* head with two fully-connected layers and two 0.5 dropout layers was used,
  and all details were important:
  features from roi pooling were very high-dimentional (more than 13k),
  first layer reduced this to 1024, and second layer
  performed final classification. Addng more layers or removing intermediate
  bottleneck reduced quality.
* bigger backbones made a big difference, best model was the largest
  that could fit into 2080ti with a reasonable batch size:
  ``resnext101_32x8d_wsl`` from https://github.com/facebookresearch/WSL-Images
* in order to train ``resnext101_32x8d_wsl`` on 2080ti, mixed precision training
  was required along with freezing first convolution and whole layer1
  (as I learned from Arthur Kuzin, this is a trick used in mmdetection:
  https://github.com/open-mmlab/mmdetection/blob/6668bf0368b7ec6e88bc01aebdc281d2f79ef0cb/mmdet/models/backbones/resnet.py#L460)
* another trick for reducing memory usage and making it train faster with
  cudnn.benchmark was limiting and bucketing number of targets in one batch.
* model was very sensitive to hyperparameters such as crop size and shape
  and batch size (and I had a bug in gradient accumulation).
* SGD with momentum performed significantly better than Adam, cosine schedule
  was used, weight decay was also quite important.
* quite large scale and color augmentations were used: hue/saturation/value,
  random brighness, contrast and gamma, all from Albumentations library.
* TTA (test-time-augmentation) of 4 different scales was used.
* ``resnext101_32x8d_wsl`` took around 15 hours to train on one 2080ti.

Best single model without pseudolabelling obtained public LB score of 0.935,
although score varied quite a lot between folds,
most folds were in 0.925 - 0.930 range.
A blend of ``resnet152`` and ``resnext101_32x8d_wsl`` models across all folds
scored 0.941 on the public LB.

Overall, many improvement are possible here, from just using bigger models
and freezing less layers, to more work on training schedule, augmentations,
etc.

See ``kuzushiji.classify.main`` for the training script,
``kuzushiji.classify.models`` for the models,
and ``kuzushiji.classify.dataset`` for the dataset and augmentations.

Pseudolabelling
---------------

Pseudolabelling is a technique where we take confident predictions of our model
on test data, and add this to train dataset. Even though the model is already confident
in such predictions, they are still useful and improve quality, because
they allow the model to adapt better to different domain, as each book
has it's own character and paper style, each author has different writing,
etc.

Here the simplest approach was chosen: most confident predictions were used
for all test set, instead of splitting it by book. Top 80% most confident
predictions from the blend were used, having accuracy >99% according to
validation. Next, two kinds of models were trained
(all based on ``resnext101_32x8d_wsl``):

- models from previous step fine-tuned for 5 epochs
  (compared to 50 epochs for training from scratch) with starting learning
  rate 10x smaller than initial learning rate.
- models trained from scratch with default settings.

In both cases, models used both train and test data for training.
Best fine-tuned model scored 0.938 on the public LB.
From-scratch models were not submitted separately but from their contribution
to the ensemble, they could be even better.

See ``kuzushiji.classify.pseudolabel`` for creation of test targets.

Second level model
------------------

A simple blend worked already quite well, giving 0.943 public LB
(without pseudolabelled from-scratch models). Adjusting coefficients of the
models didn't improve the validation score, even though ``resnext101_32x8d_wsl``
models were noticeably better.

Since all models were trained across all folds, it was possible to train
a second level model, a blend of LightGBM and XGBoost.
This model was inspired by Pavel Ostyakov's solution to
Cdiscount’s Image Classification Challenge, which was a classification
problem with 5k classes:
https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/45733

Each of 4 model kinds from classification contributed
classes and scores of top-3 predictions as features. Also max overlap
with other bboxes was added. Then for each of all classes in top-3 predictions,
and for a ``seg_fp`` class, we created one row with an extra feature ``candidate``,
which had a class as a value, and the target is binary: whether this candidate
class was a true class which should be predicted. Then for each
top-3 class, we added an extra binary feature which tells whether this class is
a candidate class.

Here is a simplified example with 1 model and top-2 predictions,
all rows created for one character prediction (``seg_fp`` was encoded as -1,
``top0_s`` means ``top0_score``, ``top0_is_c`` means ``top0_is_candidate``)::

    top0_cls  top1_cls  top0_s  top1_s  candidate  top0_is_c  top1_is_c  y
    83        258       15.202  7.1246  83         True       False      True
    83        258       15.202  7.1246  258        False      True       False
    83        258       15.202  7.1246  -1         False      False      False

XGBoost and LighGBM models were trained across all folds, and then blended.
It was better to first apply models to fold predictions on test and then
blend them.

Such blend gives 0.949 on public LB.

I'm extremely bad at tuning such models, so there may be more improvements
possible. Adjusting ``seg_fp`` ratio was tried and provided some boost on
validation but didn't work on public LB.

See ``kuzushiji.classify.level2_features`` where main features are created,
and ``kuzushiji.classify.level2`` where model are trained.

Discarded ideas
---------------

* language model: a simple bi-LSTM language model was trained, but it achieved
  log loss of only ~4.5, while image-base model was at ~0.5, so it seemed
  that it would provide very little benefit.
* kNN/metric learning: it's possible to use activations before the last layer as features,
  extract them from train and test, and then at inference time look
  closest (by cosine distance) example from train. This gave a minor boost
  over classification for single models,
  but inference time was quite high even with all optimizations,
  blending was less clear, so this was discarded.

Hardware and libraries
----------------------

Almost all models were trained on my home server with one 2080ti.
``resnet152`` classification models were trained on GCP with P100 GPUs as they
required 16 GB of memory and I had some GCP credits. A few models towards
the end were trained on vast.ai.

All models are written with pytorch, detection models are based on torchvision.
Apex is used for mixed precision training, and Albumentations for
augmentations.

License
=======

License is MIT.
Files under ``detection`` are taken from torchvision with minor modifications,
which is licensed under BSD-3-Clause. Also files in ``kuzushiji.segment``
are based on detection reference from torchvision under the same license.
