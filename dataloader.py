from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import pandas as pd
from collections import namedtuple
from matplotlib import colors
import matplotlib.pyplot as plt
import random

n_class    = 27

# a label and all meta information
Label = namedtuple( 'Label' , [
    'name'        , 
    'level3Id'    , 
    'color'       , 
    ] )

labels = [
    #       name                     level3Id  color
    Label(  'road'                 ,    0  , (128, 64,128)  ),
    Label(  'drivable fallback'    ,    1  , ( 81,  0, 81)  ),
    Label(  'sidewalk'             ,    2  , (244, 35,232)  ),
    Label(  'non-drivable fallback',    3  , (152,251,152)  ),
    Label(  'person/animal'        ,    4  , (220, 20, 60)  ),
    Label(  'rider'                ,    5  , (255,  0,  0)  ),
    Label(  'motorcycle'           ,    6  , (  0,  0,230)  ),
    Label(  'bicycle'              ,   7  , (119, 11, 32)  ),
    Label(  'autorickshaw'         ,   8  , (255, 204, 54) ),
    Label(  'car'                  ,   9  , (  0,  0,142)  ),
    Label(  'truck'                ,  10 ,  (  0,  0, 70)  ),
    Label(  'bus'                  ,  11 ,  (  0, 60,100)  ),
    Label(  'vehicle fallback'     ,  12 ,  (136, 143, 153)),  
    Label(  'curb'                 ,   13 ,  (220, 190, 40)),
    Label(  'wall'                 ,  14 ,  (102,102,156)  ),
    Label(  'fence'                ,  15 ,  (190,153,153)  ),
    Label(  'guard rail'           ,  16 ,  (180,165,180)  ),
    Label(  'billboard'            ,   17 ,  (174, 64, 67) ),
    Label(  'traffic sign'         ,  18 ,  (220,220,  0)  ),
    Label(  'traffic light'        ,  19 ,  (250,170, 30)  ),
    Label(  'pole'                 ,  20 ,  (153,153,153)  ),
    Label(  'obs-str-bar-fallback' , 21 ,  (169, 187, 214) ),  
    Label(  'building'             ,  22 ,  ( 70, 70, 70)  ),
    Label(  'bridge/tunnel'        ,  23 ,  (150,100,100)  ),
    Label(  'vegetation'           ,  24 ,  (107,142, 35)  ),
    Label(  'sky'                  ,  25 ,  ( 70,130,180)  ),
    Label(  'unlabeled'            ,  26 ,  (  0,  0,  0)  ),
]   

cmap = colors.ListedColormap([tuple(num/255 for num in label[2]) for label in labels])

class AddGaussianNoise(object):
    def __init__(self, mean=0, std=1, prob = 0.5):
        self.std = std
        self.mean = mean
        self.prob = prob
        
    def __call__(self, tensor):
        if random.random() < self.prob :
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor
    
class RandomHorizontalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob
        
    def __call__(self, img, label):
        if random.random() < self.prob :
            img, label =  transforms.functional.hflip(img), transforms.functional.hflip(label)
        return img, label
    
"""class RandomHorizontalCrop(object):
    def __init__(self, prob = 0.25):
        self.prob = prob
        
    def __call__(self, pic, label):
        if random.random() < self.prob :
            width = pic.size(-1)
            #NEED TO FINISH THIS
            """


class IddDataset(Dataset):

    def __init__(self, csv_file, n_class=n_class, transforms_=None, mode = 'test'):
        self.data      = pd.read_csv(csv_file)
        self.n_class   = n_class
        self.mode = mode
        
        self.resize = transforms.Compose([transforms.Resize(256, interpolation=2), transforms.CenterCrop(256)])
        
        self.random_flip = RandomHorizontalFlip(0.5)
        
        #self.add_noise = AddGaussianNoise(0, 0.05, 0.5)
        
        self.normalize = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_name = self.data.iloc[idx, 0]
        label_name = self.data.iloc[idx, 1]
        
        img = Image.open(img_name).convert('RGB')
        label = Image.open(label_name)
        
        img, label = self.resize(img), self.resize(label)
        
        if self.mode == 'train':
            img, label = self.random_flip(img, label)
        
        #label2 = (np.asarray(cmap(label)))*255
        #print(label2.shape)
        #PIL_image = Image.fromarray(label2.astype('uint8'))
        #display(PIL_image)
        
        img = np.asarray(img) / 255. # scaling [0-255] values to [0-1]
        label = np.asarray(label)
        
        img = self.normalize(img).float() # Normalization
        label = torch.from_numpy(label.copy()).long() # convert to tensor

        # create one-hot encoding
        h, w = label.shape
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1
        
        return img, target, label