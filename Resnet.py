import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np
print(torch.cuda.is_available())
import warnings

warnings.filterwarnings("ignore")



class block(nn.Module):
  def __init__(self,in_channels,out_channels,identity_downsample=None,stride=1):
    super(block,self).__init__()
    self.expansion=4
    self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv3 = nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,stride = 1,padding=0)
    self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
    self.relu = nn.ReLU()
    self.identity_downsample = identity_downsample

  def forward(self,x):
    identity=x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)
    if (self.identity_downsample !=None):
      identity = self.identity_downsample(identity)
    x += identity
    x = self.relu(x)
    return x
    

class resnet(nn.Module):
  def __init__(self,block,layers,img_channels,n_classes):
    super(resnet,self).__init__()
    self.in_channels=64
    self.conv1 = nn.Conv2d(img_channels,64,kernel_size=7,stride=2,padding=3)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    #resnet layers
    self.resnet1 = self._make_layer(block,layers[0],64,1)
    self.resnet2 = self._make_layer(block,layers[1],128,2)
    self.resnet3 = self._make_layer(block,layers[2],256,2)
    self.resnet4 = self._make_layer(block,layers[3],512,2)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512*4,n_classes)

  def forward(self,x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.resnet1(x)
    x = self.resnet2(x)
    x = self.resnet3(x)
    x = self.resnet4(x)

    x = self.avgpool(x)
    x = x.reshape(x.shape[0],-1)
    x = self.fc(x)
    return x



  def _make_layer(self,block,num_residual_blocks,out_channels,stride):
    identity_downsample=None
    layers=[]

    if stride!=1 or self.in_channels != out_channels*4:
      identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,out_channels*4,kernel_size=1,stride = stride),
                                          nn.BatchNorm2d(out_channels*4))
      
    layers.append(block(self.in_channels,out_channels,identity_downsample,stride))
    self.in_channels=out_channels*4
    for i in range(num_residual_blocks):
      layers.append(block(self.in_channels,out_channels))

    return nn.Sequential(*layers)



model = resnet(block,[3,4,6,3],3,2)
value = 0
for i in model.parameters():
  value += i.numel()
  print(i.shape)
print(value)

print(model)

    