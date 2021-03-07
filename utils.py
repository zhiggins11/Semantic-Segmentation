import torch
#################NEEED TO DEAL WITH UNLABELED CLASS
def intersections_and_unions(pred, target):
    """
    pred: Tensor of shape (batch_size, pic_length, pic_width, number_of_classes).  The value of pred[n][i][j][k] should be the predicted
          probability that the (i,j) pixel of the n-th example belongs to the k-th class
    target: Tensor of shape (batch_size, pic_length, pic_width, number_of_classes).  target[i][j][k] should be 1 if the (i,j) pixel
          belongs to the k-th class and 0 otherwise
    
    output: list containing the intersection-over-union values for each class for the given prediction and target
    """
    inter_and_union = torch.zeros(2,n_class)
    pred_labels = torch.argmax(pred, dim = 1) #need to check shape of this
    target_labels = torch.argmax(target, dim = 1)
    for cls in range(n_class):
        inter_and_union[0][cls] = torch.sum(torch.mul(torch.eq(pred_labels, cls), torch.eq(target_labels, cls)))
        inter_and_union[1][cls] = torch.sum(torch.eq(pred_labels, cls)) + torch.sum(torch.eq(target.labels, cls)) - inter_and_union[0][cls]
        #if union == 0:
        #    ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        #else:
        #    ious.append(intersection/union)
    return inter_and_union


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