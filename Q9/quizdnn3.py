# -*- coding: utf-8 -*-
"""QuizDNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EHHYAu93Hhzxh9B3lwI25BepH1NqM-KN
"""

from __future__ import print_function
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 30*30*64 RF=3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 28*28*128  RF=5

         # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value)
        ) # output_size = 26*26*256  RF=7
        # TRANSITION BLOCK 1
        #max pooling
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 13*13*256  RF=8
         # output_size = 26
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 13*13*64  RF=7

        # convolution block 5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 13*13*128  RF=11

         # convolution block 6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value)
        ) # output_size = 13*13*256  RF=15

        #max pooling
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 7*7*256  RF=17
         # output_size = 26
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 6*6*64  RF=17

        # convolution block 8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 6*6*128  RF=25

        # convolution block 9
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            
        ) # output_size = 6*6*256  RF=33

        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1*1*256  

        #FC layer
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            
        ) # output_size = 1*1*10  


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x1 = self.convblock1(x)
        x2 = self.convblock2(x1)
        x3 = self.convblock3(x2)
        x4 = self.pool1(x3)
        x5 = self.convblock4(x4)
        x6 = self.convblock5(x5)
        x7 = self.convblock6(x6)
        x8 = self.pool2(x7)
        x9 = self.convblock7(x8)
        x10 = self.convblock8(x9)
        x11 = self.convblock9(x10)
        x12 = self.gap(x11)        
        x13 = self.convblock10(x12)

        x14 = x13.view(-1, 10)
        return F.log_softmax(x14, dim=-1)

train_losses_normal = []
test_losses_normal = []
train_acc_normal = []
test_acc_normal = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    
    train_losses_normal.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Accuracy={100*correct/processed:0.2f}')
    train_acc_normal.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses_normal.append(test_loss)

    print('\nTest set: Avg loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc_normal.append(100. * correct / len(test_loader.dataset))