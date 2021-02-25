import torch
#################NEEED TO DEAL WITH UNLABELED CLASS
def iou(pred, target):
    """
    pred: Tensor of shape (batch_size, pic_length, pic_width, number_of_classes).  The value of pred[i][j][k] should be the predicted
          probability that the (i,j) pixel belongs to the k-th class
    target: Tensor of shape (batch_size, pic_length, pic_width, number_of_classes).  target[i][j][k] should be 1 if the (i,j) pixel
          belongs to the k-th class and 0 otherwise
    
    output: list containing the intersection-over-union values for each class for the given prediction and target
    """
    ious = []
    pred_labels = torch.argmax(pred, dim = 1)
    target_labels = torch.argmax(target, dim = 1)
    for cls in range(n_class):
        intersection = torch.sum(torch.mul(torch.eq(pred_labels, cls), torch.eq(target_labels, cls)))
        union = torch.sum(torch.eq(pred_labels, cls)) + torch.sum(torch.eq(target.labels, cls)) - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection/union)
    return ious


def pixel_acc(pred, target):
    """
    pred: Tensor of shape (batch_size, pic_length, pic_width, number_of_classes).  The value of pred[i][j][k] should be the 
          predicted probability that the (i,j) pixel belongs to the k-th class
    target: Tensor of shape (batch_size, pic_length, pic_width, number_of_classes).  target[i][j][k] should be 1 if the (i,j) pixel
          belongs to the k-th class and 0 otherwise
          
    output: pixel accuracy of the predicted labels compared to the target
    """
    pred_labels = torch.argmax(pred, dim = 1)
    target_labels = torch.argmax(target, dim =1)
    return torch.mean(torch.eq(pred_labels, target_labels))