from torchvision import utils
from basic_fcn import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = IddDataset(csv_file='train.csv')
val_dataset = IddDataset(csv_file='val.csv')
test_dataset = IddDataset(csv_file='test.csv')


train_loader = DataLoader(dataset=train_dataset, batch_size= 32, num_workers= 0, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 32, num_workers= 0, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= 32, num_workers= 0, shuffle=False)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.xavier_uniform_(m.bias.data)        

epochs = 30
weights = torch.load('weights.pt')
criterion = nn.CrossEntropyLoss(weights).to(device)
fcn_model = FCN(n_class=n_class).to(device)
fcn_model.apply(init_weights)

optimizer = optim.Adam(fcn_model.parameters(), lr=.001)

        
def train():
    for epoch in range(epochs):
        ts = time.time()
        training_loss = 0
        counter = 0
        for iter, (X, tar, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            outputs = fcn_model(X)
            loss = criterion(outputs, Y)
            training_loss += loss.item()
            counter += 1
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, 'best_model')
        
        training_loss /= counter

        val(epoch)
        fcn_model.train()
    


def val(epoch):
    fcn_model.eval()
    ts = time.time()
    softmax = nn.Softmax(3)
    validation_loss = 0
    ious = torch.zeros(n_class)
    pixel_accuracy = torch.zeros(256, 256)
    counter = 0
    
    with torch.no_grad():
        for iter (X, target, Y) in enumerate(val_loader):
            X, target, Y = X.to(device), target.to(device), Y.to(device)
            outputs = fcn_model(X)
            loss = criterion(outputs, Y)
            validation_loss += loss.item()
            
            predictions = softmax(outputs)
            ious += iou(predictions, target)
            pixel_accuracy += pixel_acc(predictions, target)
            counter += 1
                
    print("Finish epoch {} validation, time elapsed {}".format(epoch, time.time() - ts))
    return validation_loss/counter, ious/counter, pixel_accuracy/counter  
            
    # Don't forget to put in eval mode !
    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    
def test():
	fcn_model.eval()
    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    
if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()