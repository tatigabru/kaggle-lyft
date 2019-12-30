import torch 
import numpy as np
import torch.nn.functional as F

SMOOTH = 1e-6
classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]


def iou_pytorch(pred, target, n_classes = 9, print_table = True):
    """
    PyTorch IoU implementation 
    from: 
    """   
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    pred = torch.Tensor(pred).cuda().round()

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection    
    
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    if print_table:        
        print(f'classes: {classes}')
        print(f'ious: {ious}, mean {np.nanmean(ious)}')     
      
    return np.array(ious), np.mean(ious)


def precision_at(threshold, iou):
    """ Get true positive, false positive, false negative at iou threshold
    """
    matches = iou >= threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def binarize_predictions(prediction):
    """Binarise output masksfor one-hot classes"""
    num_classes = NUM_CLASSES
    #prediction = prediction.cpu().numpy()
    output = np.rint(prediction).astype(np.uint8)
    output_one_hot = (output[:, :, None] -1 == np.arange(num_classes)[None, None, :]).astype(np.uint8)
    
    return output_one_hot


def iou_numpy(outputs: np.array, labels: np.array, classes = classes, num_classes = 9, print_table = False):
    """
    Multiclass IoU 
    Numpy version
    """
    SMOOTH = 1e-6
    ious = []
    #outputs = outputs.squeeze(1)
    outputs = np.rint(outputs).astype(np.uint8)
    for num in range(1, num_classes+1):
        intersection = ((outputs==num) * (labels==num)).sum()
        union = (outputs==num).sum() + (labels==num).sum() - intersection
        if union == 0: 
            ious.append(float('nan'))  # if there is no class in ground truth, do not include in evaluation
        else:  
            ious.append((intersection + SMOOTH) / (union + SMOOTH))    
    #thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    if print_table:        
        print(f'classes: {classes}')
        print(f'ious: {ious}, mean {np.nanmean(ious)}')         
    
    return ious, np.nanmean(ious)    


def test_iou_pytorch():
    a = np.array([[1.1, 1.4, 5.3, 0],
                 [2.1, 4.6, 2.3, 0], 
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]) 

    b = np.array([[1, 1, 5, 4],
                 [2, 2, 1, 1],
                 [7, 3, 3, 3],
                 [0, 0, 0, 0]])
    a_rounded = torch.Tensor(a).cuda().round().type(torch.int8)
    a_rounded= a_rounded.dtype(torch.int8)
    print(f'a {a}\n b {b} \n a_rounded {a_rounded}, a_rounded.dtype {a_rounded.dtype}') 
    ios, iou = iou_pytorch(a, b)
    print(f"IoUs: {ious} mean IoU: {iou}")


def test_iou_multiclass():
    A = torch.tensor([
    [[1.1, 1.2, 2.3, 2.1],
     [1, 1, 2, 3],
     [1, 1, 3, 3]]
    ]) 
    B = torch.tensor([
    [[1, 1, 2, 2],
     [1, 1, 2, 2],
     [1, 3, 3, 2]]
    ])
    A = torch.Tensor(A).cuda().round().type(torch.int8)
    A_oh = F.one_hot(A)
    B_oh = F.one_hot(B)
    print(f'a {A_oh},\n b {A_oh}')

    int_AB = A_oh & B_oh
    union_AB = A_oh | B_oh
    iou = int_AB.sum(1).sum(1).type(torch.float32) / union_AB.sum(1).sum(1).type(torch.float32)
    print(iou[:, 1:])


def test_iou_sample():
    a = np.array([[1.1, 1.4, 5.3, 0],
                 [2.1, 4.6, 2.3, 0], 
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]) 

    b = np.array([[1, 1, 5, 4],
                 [2, 2, 1, 1],
                 [7, 3, 3, 3],
                 [0, 0, 0, 0]]) 
    print(f'a {a},\n b {b}')             

    a = np.rint(a).astype(np.uint8)
    ious, iou = iou_numpy(a,b) 
    rint(f"IoUs: {ious} mean IoU: {iou}")


def main():
    test_iou_sample() 
    test_iou_multiclass()
    test_iou_pytorch()   
    


if __name__ == "__main__":
    main()        


