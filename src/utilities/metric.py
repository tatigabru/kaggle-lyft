"""
Tati Gabru
and
https://www.kaggle.com/aglotero/another-iou-metric

"""
import numpy as np
from functools import partial
import torch
import sys
sys.path.append('/home/user/challenges/lyft/lyft_repo')

from configs import NUM_CLASSES


def precision_at(threshold, iou):
    """ Get true positive, false positive, false negative at iou threshold
    """
    matches = iou >= threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def get_iou(labels, y_pred):
    # https://www.kaggle.com/aglotero/another-iou-metric
    skip_class = False
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    union =  area_true + area_pred - intersection
    if np.sum(labels)==0 and np.sum(y_pred)==0:
        skip_class = True
    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union
    
    return iou, skip_class


def map_per_class(labels, y_pred, print_table=False):
    """Get average_precision over thresholds
    From:
    https://www.kaggle.com/aglotero/another-iou-metric
    Args:
        
    """
    iou, skip_class = get_iou(labels, y_pred)
    if skip_class == True:
        if print_table:
            print('No class and false positives')
        return None
    else:    
        # Loop over IoU thresholds
        prec = []
        if print_table:
            print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            if (tp + fp + fn) > 0:
                p = tp / (tp + fp + fn)
            else:
                p = 0
            if print_table:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)
        if print_table:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

        return np.mean(prec)    


def map_sample(target, prediction):
    """Calculates IoU for a single sample
    Args: 
        target: true mask 
        output: model output
    """     
    epsilon = 1e-7
    num_classes = NUM_CLASSES

    # move all to numpy
    #output = prediction.detach().cpu().numpy()
    #target = target.detach().cpu().numpy()
    output = np.rint(prediction).astype(np.uint8)

    # one-hot masks encoding
    target_one_hot = (target[:, :, None] -1 == np.arange(num_classes)[None, None, :]).astype(np.uint8)
    output_one_hot = (output[:, :, None] -1 == np.arange(num_classes)[None, None, :]).astype(np.uint8)
    
    mAP = []
    for class_num in range(num_classes):
        labels = target_one_hot[:, :, class_num]
        y_pred = output_one_hot[:, :, class_num]
        iou, skip_class = get_iou(labels, y_pred)
        map_class = map_per_class(labels, y_pred, True)
        if map_class is not None:        
            mAP.append(map_class)
    mAP = np.asarray(mAP) 
    #print(mAP)   

    return np.mean(mAP)   


def test_map_sample():
    a = np.array([[1.1, 1.4, 5.3, 4.7],
                 [2.1, 4.6, 1.3, 8.9], 
                 [7.3, 3.4, 3.3, 3.2],
                 [7.1, 7.6, 8.3, 8.9]]) 

    b = np.array([[1, 1, 5, 4],
                 [2, 2, 3, 9],
                 [7, 3, 3, 3],
                 [7, 2, 8, 9]]) 
    print(f'a {a},\n b {b}')                 

    mAP = map_sample(b, a)
    print(f"mean AP: {mAP}")
    

def binarize_predictions(prediction):
    """Binarise output masksfor one-hot classes"""
    num_classes = NUM_CLASSES
    #prediction = prediction.detach().cpu().numpy()
    output = np.rint(prediction).astype(np.uint8)
    output_one_hot = (output[:, :, None] -1 == np.arange(num_classes)[None, None, :]).astype(np.uint8)
    
    return output_one_hot


def test_binarize_predictions():
    a = np.array([[0.1, 7.4, 5.3, 3.7],
                 [2.1, 4.6, 1.3, 8.9]])    
    print(np.arange(9))
    print(binarize_predictions(a))


def main():
    test_binarize_predictions()
    test_map_sample()    


if __name__ == "__main__":
    main()        
