import os
import sys
sys.path.append('/home/user/challenges/lyft/lyft_repo/src')

import segmentation_models_pytorch as smp
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchsummary
import torchvision
from configs import CLASSES, NUM_CLASSES
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# my imports 
from unet import UNet


def get_spm_model(encoder='resnet50', num_classes=NUM_CLASSES+1):
    """Get model from qubvel libruary
        create segmentation model with pretrained encoder
    Args: 
        encoder: 'resnext101_32x8d', 'resnet18', 'resnet50', 'resnet101' ...
    """
    model = smp.Unet(
        encoder_name=encoder, 
        encoder_weights='imagenet',
        classes=num_classes, 
        activation='softmax',)
    # Optional, for multi GPU training and inference
    #model = nn.DataParallel(model)
    return model


def get_fpn_model(encoder='resnet50', num_classes=NUM_CLASSES+1):
    """Get model from qubvel libruary
        create segmentation model with pretrained encoder
    Args: 
        encoder: 'resnext101_32x8d', 'resnet18', 'resnet50', 'resnet101' ...
    """
    model = smp.FPN(
        encoder_name=encoder, 
        encoder_weights='imagenet',
        classes=num_classes, 
        activation='softmax',)
    # Optional, for multi GPU training and inference
    #model = nn.DataParallel(model)
    return model    


def get_unet_two_head_model(encoder='resnet50', num_classes=NUM_CLASSES+1, dropout=0.5):
    """Get model from qubvel libruary
        create segmentation model with two heads: 
            segmentation andclassification
    Args: 
        encoder: 'resnext101_32x8d', 'resnet18', 'resnet50', 'resnet101' ...
        dropout: dropout for the classification head as it overfits faster
    """
    model = smp.Unet(
        encoder_name=encoder, 
        encoder_weights='imagenet',
        in_channels = 3,
        classes=num_classes, 
        activation='softmax',
        aux_params=dict(classes=num_classes, activation='softmax', dropout=dropout))
    # Optional, for multi GPU training and inference
    #model = nn.DataParallel(model)
    return model


def get_fpn_two_head_model(encoder='resnet50', num_classes=NUM_CLASSES+1, dropout=0.5):
    """Get model from qubvel libruary
        create segmentation model with two heads: 
            segmentation andclassification
    Args: 
        encoder: 'resnext101_32x8d', 'resnet18', 'resnet50', 'resnet101' ...
    """
    model = smp.FPN(
        encoder_name=encoder, 
        encoder_weights='imagenet',
        in_channels = 3,
        classes=num_classes, 
        activation='softmax',
        aux_params=dict(classes=num_classes, activation='softmax', dropout = dropout))
    # Optional, for multi GPU training and inference
    #model = nn.DataParallel(model)
    return model


def get_unet_model(in_channels=3, num_output_classes=2):
    """Get classical unet model        
    Args: 
        in_channels: number of input channels
        num_output_classes: number of output classes
    """
    model = UNet(in_channels=in_channels, n_classes=num_output_classes, wf=5, depth=4, padding=True, up_mode='upsample')    
    # Optional, for multi GPU training and inference
    #model = nn.DataParallel(model)
    return model

      
def get_maskrcnn_model(num_classes=NUM_CLASSES):
    """Get Mask-RCNN model from torchvision libruary        
    Args: 
        num_classes: number of output classes
    """
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def print_model_summary(model):
    """Prints all layers and dims of the pytorch net"""    
    torchsummary.summary(model, (1, 512, 512))
