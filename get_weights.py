from dataloader import *
import torch
import os
cwd = os.getcwd()
print(cwd)

train_dataset = IddDataset(csv_file='train.csv')
train_loader = DataLoader(dataset=train_dataset, batch_size= 32, num_workers= 0, shuffle=True)
counter = torch.zeros(27) #there are 27 classes

for iter, (X, tar, Y) in enumerate(train_loader):
    print(Y.shape)
    for pixel in Y:
        counter[pixel] += 1

average = torch.mean
weights = average/counter

torch.save(weights, 'weights.pt')

#Might want to weight unlabeled class differently?
        
