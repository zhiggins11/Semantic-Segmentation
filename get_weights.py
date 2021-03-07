from dataloader import *
import torch

from torchvision import utils
from basic_fcn import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dataset = IddDataset(csv_file='train.csv')
train_loader = DataLoader(dataset=train_dataset, batch_size= 4, num_workers= 0, shuffle=True)
counts = torch.zeros(27).to(device) #there are 27 classes

for iter, (X, tar, Y) in enumerate(train_loader):
    tar = tar.to(device)
    counts += torch.mean(tar, (0,2,3))
    if iter % 100 == 0:
        print(iter)
average = torch.mean(counts)
weights = average * counts.pow_(-1)

torch.save(weights, 'weights.pt')

#Might want to weight unlabeled class differently?

 