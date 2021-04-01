import torch
import torch.nn as nn
from dataloader import *

def iou(pred, target):
    """
    pred: Tensor of shape (batch_size, num_of_classes, pic_length, pic_width).  The value of pred[i][j][k][l] should be the predicted
          probability that the (k,l) pixel of the i-th batch example belongs to the j-th class
    target: Tensor of shape (batch_size, num_of_classes, pic_length, pic_width).  target[i][j][k][l] should be 1 if the (k,l) pixel 
          of the i-th batch example belongs to the j-th class and 0 otherwise
    
    output: inter_and_union has shape (2, num_classes), where inter_and_union[0][i] is the size of the intersection for class i (for
          the given batch), and inter_and_union[1][i] is the size of the union for class i.
    """
    inter_and_union = torch.zeros(2,27)
    pred_labels = torch.argmax(pred, dim = 1)
    target_labels = torch.argmax(target, dim = 1)
    for cls in range(n_class):
        inter_and_union[0][cls] = torch.sum(torch.mul(torch.eq(pred_labels, cls), torch.eq(target_labels, cls)))
        inter_and_union[1][cls] = torch.sum(torch.eq(pred_labels, cls)) + torch.sum(torch.eq(target_labels, cls)) - inter_and_union[0][cls]
    return inter_and_union

def pixel_acc(pred, target):
    """
    pred: Tensor of shape (batch_size, num_of_classes, pic_length, pic_width).  The value of pred[i][j][k][l] should be the predicted
          probability that the (k,l) pixel of the i-th batch example belongs to the j-th class
    target: Tensor of shape (batch_size, num_of_classes, pic_length, pic_width).  target[i][j][k][l] should be 1 if the (k,l) pixel 
          of the i-th batch example belongs to the j-th class and 0 otherwise
          
    output: pixel accuracy of the predicted labels compared to the target
    """
    pred_labels = torch.argmax(pred, dim = 1)
    target_labels = torch.argmax(target, dim =1)
    return torch.mean(torch.eq(pred_labels, target_labels).float())

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        pred: Tensor of shape (batch_size, num_of_classes, pic_length, pic_width).  pred should be the raw output from the network which
            we will pass through a softmax.
        target: Tensor of shape (batch_size, num_of_classes, pic_length, pic_width).  target[i][j][k][l] should be 1 if the (k,l) pixel 
          of the i-th batch example belongs to the j-th class and 0 otherwise
        """
        softmax = nn.Softmax(1)
        pred = softmax(pred)
        #intersection = 2*torch.sum(pred * target, (0,2,3))
        #union = torch.sum(pred, (0,2,3)) + torch.sum(target, (0,2,3))
        #loss = (torch.sum(torch.div(intersection,union))/pred.size(1) - intersection[26]/union[26])/26 #We don't care about the unlabeled class
        numerator = 2 * torch.sum(pred*target)
        denominator = torch.sum(pred) + torch.sum(target)
        loss = numerator/denominator
        return 1 - loss
        
        
