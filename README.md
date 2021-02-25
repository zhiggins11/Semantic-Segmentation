# Fully Convolutional Networks for Semantic Segmentation

##Background
This project has several fully convolutional networks which can be used to perform semantic segmentation.  It uses a subset of the India Driving Dataset to train, validate, and test the models.  Links to the images in the training, validation, and test sets are found in the files train.csv, val.csv, and test.csv, respectively.

##files

1. basic_fcn.py contains the FCN class, which is a baseline fully convolutional network.  
2. more_fcn.py contains the FCN_8 and FCN_16 classes, which are other FCNs we designed.
3. transfer_fcn.py contains the Transfer_FCN class, which is like the FCN class but uses the convolution layers of ResNet-18 for its encoder.
4. U_net.py contains the U_Net class, which is an FCN with architecture based on the paper https://arxiv.org/pdf/1505.04597.pdf.
5. starterWeighted.py contains code for training and evaluating the U_Net model.  It can be changed (see "Training and evaluating a model" below) to train and evaluate the FCN, FCN_8, or FCN_16 models as well
6. starterTransfer.py contains code for training and evaluating the Transfer_FCN model.  
7. util.py contains functions for calculating the dice coefficient, IoU, and pixel accuracy
8. dataloader.py contains functions for loading, preprocessing, and transforming the images
9. train.csv contains links to the images in the training set
10. val.csv contains links to the images in the validation set
11. test.csv contains links to the images in the test set

##Training and evaluating a model
Running the file starterWeighted.py as is will create a U_Net model, train it for 50 epochs, and save lists that contain
1. The training loss for each epoch (trainL.txt)
2. The validation loss for each epoch (validL.txt)
3. The validation pixel accuracy for each epoch (validA.txt)
4. The validation Intersection over Unions (IoUs) for each category in the dataset, for each epoch (categoryAccV.txt)
5. The average of the validation IoUs over the categories, for each epoch (avgIoUV.txt)
6. The first picture of the test dataset, its pixel labelings, and the predicted category for each pixel by the trained model (WeightMap.png)
7. A graph of the training and validation losses throughout training (Weightimplementation.png)
8. IoU for each category, pixel accuracy, and average IoU on the test set (TestfileWeight.txt)

To train and evaluate a different one of our models, simply replace "U_Net" in line 57 with one of "FCN", "FCN_8", or "FCN_16".
To train and evaluate the transfer learning model, run starterTransfer.py instead.

