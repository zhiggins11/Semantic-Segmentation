import torch
import torch.nn as nn

class DoubleConv2d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1   = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bnd1    = nn.BatchNorm2d(out_channels)
        self.conv2   = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bnd2    = nn.BatchNorm2d(out_channels)
        
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bnd1(self.relu(self.conv1(x)))
        x = self.bnd2(self.relu(self.conv2(x)))
        
        return x

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        self.conv1   = DoubleConv2d(3, 64)
        self.conv2   = DoubleConv2d(64,128)
        self.conv3   = DoubleConv2d(128,256)
        self.conv4   = DoubleConv2d(256,512)
        self.conv5   = DoubleConv2d(512,1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6   = DoubleConv2d(1024,512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7   = DoubleConv2d(512,256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8   = DoubleConv2d(256,128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9   = DoubleConv2d(128,64)
        self.classifier = nn.Conv2d(64,n_class, kernel_size=1)
        
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        y = self.conv5(self.maxpool(x4))
        y = self.deconv1(y)
        y = torch.cat((x4,y),1)
        y = self.deconv2(self.conv6(y))
        y = torch.cat((x3,y),1)
        y = self.deconv3(self.conv7(y))
        y = torch.cat((x2,y),1)
        y = self.deconv4(self.conv8(y))
        y = torch.cat((x1,y),1)
        y = self.classifier(self.conv9(y))
        
        return y