# Semantic Segmentation Using a Fully Convolutional Network

This project is work that I did for a course on deep learning at UCSD (CSE 251B).  It contains two different convolutional neural networks that I implemented - a standard fully convolutional neural network (basic_fcn.py) and [U-Net](https://arxiv.org/pdf/1505.04597.pdf) (UNet.py).

These neural networks were trained to perform semantic segmentation on images taken from car dashcams.  I used part of the [India Driving Dataset] (https://idd.insaan.iiit.ac.in/) to train, validate, and test the models.

## Visual Results

Here are some labelings generated by the trained model on images in the test set.  Each strip contains the actual image, ground truth labels for that image, and model predictions, in that order.

![test1](https://user-images.githubusercontent.com/77809548/110228807-ea01c980-7eb8-11eb-9ae4-b46b0171bee5.png)

![test2](https://user-images.githubusercontent.com/77809548/110228861-70b6a680-7eb9-11eb-867e-08d333628125.png)

![test3](https://user-images.githubusercontent.com/77809548/110228917-db67e200-7eb9-11eb-80e8-3110dbe007fe.png)

## Numerical Results

A trained model, which can be loaded from `latest_model.pt` was evaluated on the test set, and gave a pixel accuracy of 0.8134 as well as the following intersection-over-union (IoU) values on each category\
0. Road - 0.903
1. Drivable fallback - 0.416
2. Sidewalk - 0.101
3. Non-drivable fallback - 0.246
4. Person/animal - 0.151
5. Rider - 0.253
6. Motorcycle - 0.317
7. Bicycle - 0.034
8. Autorickshaw - 0.395
9. Car - 0.461
10. Truck - 0.238
11. Bus - 0.179
12. Vehicle Fallback - 0.246
13. Curb - 0.426
14. Wall - 0.221
15. Fence - 0.065
16. Guard Rail - 0.137
17. Billboard - 0.132
18. Traffic Sign - 0.021
19. Traffic Light - 0
20. Pole - 0.156
21. Obs-str-bar-fallback - 0.130
22. Building - 0.409
23. Bridge/tunnel - 0.344
24. Vegetation - 0.756
25. Sky - 0.942


## Files

`basic_fcn.py` contains the class for the basic fully convolutional network.\
`UNet.py` contains the class for U-Net.\
`dataloader.py ` contains code for loading training, validation, and test datasets.\
`latest_model.pt` contains a trained model which can be used to make predictions on test images.\
`starter.py` contains code needed to train a model.  If you would like to run this, you'll need to download (some portion of) the India Driving Dataset and save links to each of the training, validation, and test images to the files `train.csv`, `val.csv`, and `test.csv`, respectively, in your working directory.\
`utils.py` contains functions used to compute pixel accuracy and IoU for each category, as well as a DiceLoss class, which can be used to train a model using dice loss rather than cross entropy loss.\
`get_weights.py` contains code for computing weights for each of the classes.  These weights can be used to train a model using weighted cross entropy loss.


## Future Work
While the trained model has a relatively high pixel accuracy on the test set, this is likely due to several categories (road, sky, vegetation, etc.) dominating the majority of the pictures.  I would like to try to improve the model's predictions on some of the other categories by using either dice loss or weighted cross entropy loss.  Both of these loss functions have been implemented in the files above, but unfortunately I don't currently have access to computing power necessary to train these models.
