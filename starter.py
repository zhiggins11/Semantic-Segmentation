from torchvision import utils
from basic_fcn import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = IddDataset(csv_file='train.csv')
val_dataset = IddDataset(csv_file='val.csv')
test_dataset = IddDataset(csv_file='test.csv')


train_loader = DataLoader(dataset=train_dataset, batch_size= 32, num_workers= 0, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 32, num_workers= 0, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= 32, num_workers= 0, shuffle=False)

epochs = 30

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)       

criterion = nn.CrossEntropyLoss().to(device)
fcn_model = FCN(n_class=n_class).to(device)
fcn_model.apply(init_weights)

optimizer = optim.Adam(fcn_model.parameters(), lr=.001)


        
def train():
    best_val_loss = 100000
    training_losses = []
    val_losses = []
    for epoch in range(epochs):
        ts = time.time()
        training_loss = 0
        counter = 0
        for iter, (X, target, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            outputs = fcn_model(X)
            loss = criterion(outputs, Y)
            training_loss += loss.item()
            counter += 1
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch {}, iter {}, loss: {}".format(epoch + 1, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save({
            'epoch': epoch,
            'model_state_dict': fcn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, 'latest_model.pt')

        
        training_loss /= counter
        val_loss, val_iou, val_accuracy = val(epoch)
        print("Training Loss: {}".format(training_loss))
        print("Validation Loss: {}".format(val_loss))
        print("Validation IOU: {}".format(val_iou))
        print("Validation Accuracy: {}".format(val_accuracy))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict': fcn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, 'best_model.pt')

        training_losses.append(loss.item())
        val_losses.append(val_loss)
        plot_stats(training_losses, val_losses)
        fcn_model.train()
    


def val(epoch):
    fcn_model.eval()
    ts = time.time()
    softmax = nn.Softmax(1) #I had softmax(3) here before for some reason?
    validation_loss = 0
    inter_and_union = torch.zeros(n_class)
    pixel_accuracy = torch.zeros(256, 256)
    counter = 0
    
    with torch.no_grad():
        for iter, (X, target, Y) in enumerate(val_loader):
            X, target, Y = X.to(device), target.to(device), Y.to(device)
            outputs = fcn_model(X)
            loss = criterion(outputs, Y)
            validation_loss += loss.item()
            
            predictions = softmax(outputs)
            inter_and_union = iou(predictions, target)
            pixel_accuracy += pixel_acc(predictions, target)
            counter += 1
    ious = torch.div(inter_and_union[0], inter_and_union[1])
                
    print("Finish epoch {} validation, time elapsed {}".format(epoch+1, time.time() - ts))
    return validation_loss/counter, ious, pixel_accuracy/counter  
            
    
def test():
    fcn_model.eval()
    softmax = nn.Softmax(1)
    loss = 0
    inter_and_union = torch.zeros(n_class)
    pixel_accuracy = torch.zeros(256, 256)
    counter = 0
    
    with torch.no_grad():
        for iter, (X, target, Y) in enumerate(test_loader):
            X, target, Y = X.to(device), target.to(device), Y.to(device)
            outputs = fcn_model(X)
            loss += criterion(outputs, Y).item()
            
            predictions = softmax(outputs)
            inter_and_union = iou(predictions, target)
            pixel_accuracy += pixel_acc(predictions, target)
            counter += 1
    ious = torch.div(inter_and_union[0], inter_and_union[1])
    return loss/counter, ious, pixel_accuracy/counter  
            
    
def plot_stats(training_losses, val_losses):
        x_axis = np.arange(1, len(training_losses) + 1, 1)
        plt.figure()
        plt.plot(x_axis, training_losses, label="Training Loss")
        plt.plot(x_axis, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title("Training and validation losses")
        plt.show()

    
if __name__ == "__main__":
    train()