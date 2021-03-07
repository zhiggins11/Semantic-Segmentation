# Semantic Segmentation Using a Fully Convolutional Network

This is work that I did as part of a group project (with Lingxi Li, Yejun Li, Jiawen Zeng, and Yunyi Zhang) for CSE 251B (Neural Networks).  I benefited from discussions with my partners, but all code here was either given by the instructor or written by me.  Specifically, I created the model (basic_fcn.py) and wrote the code for training the model and evaluating it on the validation and tests sets (starter.py and util.py), and the instructor gave us all the code for loading the data.


This project has a basic fully convolutional neural network that can be used for semantic segmentation.  It uses a subset of the India Driving Dataset to train, validate, and test the models.

## Visual Results

Here are some test images where the model performs well.  Each strip contains the actual image, ground truth labels for that image, and model predictions, in that order.

![test1](https://user-images.githubusercontent.com/77809548/110228807-ea01c980-7eb8-11eb-9ae4-b46b0171bee5.png)

![test2](https://user-images.githubusercontent.com/77809548/110228861-70b6a680-7eb9-11eb-867e-08d333628125.png)

![test3](https://user-images.githubusercontent.com/77809548/110228917-db67e200-7eb9-11eb-80e8-3110dbe007fe.png)

Here are some test images where the model doesn't perform very well.  Each strip contains the actual image, groud truth labels, and model predictions, in that order.




## Numerical Results

The trained model, which can be loaded from `latest_model.pt` was evaluated on the test set, and gave a pixel accuracy of 0.8134 as well as the following intersection-over-union (IoU) values on each category
0. Road - 0.903
1. Drivable fallback - 0.416
2. Sidewalk - 0.101
3. Non-drivable fallback - 0.246
4. Person/animal - 0.151
5. Rider - 0.253
6. Motorcycle - 0.317
7. Bicycle - 0
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

