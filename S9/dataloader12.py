# -*- coding: utf-8 -*-
"""dataloader.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AflSpxgdQ42WJXLilCNWW1KTal3MrqS3
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import albumentations as alb
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset

class album_compose_train:
  def __init__(self):
     
        meandata, stddata=album_calculate_dataset_mean_std()
        print('mean0',meandata[0])
        print('mean1',meandata[1])
        print('mean2',meandata[2])
        print('std0',stddata[0])
        print('std1',stddata[1])
        print('std2',stddata[2])
        channel_mean=(meandata[0]+meandata[1]+meandata[2])/3.0
        print('channel mean',channel_mean)
        self.albtransform = alb.Compose([
        
        alb.HorizontalFlip(p=0.5),
        alb.Rotate(limit=(-90, 90)),
        alb.Cutout(fill_value=channel_mean*255),
        alb.Normalize((meandata[0], meandata[1], meandata[2]), (stddata[0], stddata[1], stddata[2])),
        ToTensor(),
        ])
 
  def __call__(self,img):
    img=np.array(img)
    img=self.albtransform(image=img)
    return img['image']

class album_compose_test:
  def __init__(self):
        meandata, stddata=album_calculate_dataset_mean_std()
        self.albtransform = alb.Compose([
        
        print('meandata[0]:',meandata[0]),
        alb.Normalize((meandata[0], meandata[1], meandata[2]), (stddata[0], stddata[1], stddata[2])),
        ToTensor(),
        ])
 
  def __call__(self,img):
    img=np.array(img)
    img=self.albtransform(image=img)
    return img['image']

def load_data():
  #transform_test = transforms.Compose(
   #   [transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  #transform_train = transforms.Compose([
  #    transforms.RandomCrop(32, padding=4),
  #    transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
  #    transforms.RandomRotation((-10.0, 10.0)), transforms.ToTensor(),
  #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  #])
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True,transform=album_compose_train())
  
  #alb_dataset = AlbumentationImageDataset(image_list=trainset)

  #trainloader = DataLoader(alb_dataset, batch_size= 128,
  #                                          shuffle=True, num_workers=4)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=album_compose_test())
  #testloader = DataLoader(testset, batch_size=128,
  #                                        shuffle=False, num_workers=4)

  classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  #                                         shuffle=False, num_workers=2)
  #classes = ('plane', 'car', 'bird', 'cat',
  #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  
  SEED = 1

  # CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  # For reproducibility
  torch.manual_seed(SEED)

  if cuda:
      torch.cuda.manual_seed(SEED)

  # dataloader arguments - something you'll fetch these from cmdprmt
  dataloader_args = dict(shuffle=True, batch_size=64, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=16)

  # train dataloader
  train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

  # test dataloader
  test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)
  return train_loader, test_loader, classes

  #return trainloader, testloader, classes


def album_calculate_dataset_mean_std():

    trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=ToTensor())

    testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=ToTensor())



    data = np.concatenate([trainset.data, testset.data], axis=0)

    data = data.astype(np.float32)/255.



    print("\nTotal dataset(train+test) shape: ", data.shape)



    means = []

    stdevs = []

    for i in range(3): # 3 channels

        pixels = data[:,:,:,i].ravel()

        means.append(np.mean(pixels))

        stdevs.append(np.std(pixels))



    return [means[0], means[1], means[2]], [stdevs[0], stdevs[1], stdevs[2]]