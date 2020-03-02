# -*- coding: utf-8 -*-
"""Miscellaneous.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p6Iz9ZjwN48nL_-PmD-Qgkk08Msx3EA4
"""

import plotimagefinal as pimage
import torchvision
import torch
def prtgroundtruth(test_loader,classes):
  dataiter = iter(test_loader)
  images, labels = dataiter.next()

  # print images
  pimage.imshow(torchvision.utils.make_grid(images))
  print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
  return images

def getTestAccuracy(model,images,classes,test_loader,device):
  outputs = model(images.to(device))
  _, predicted = torch.max(outputs, 1)

  print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))
  correct = 0
  total = 0
  with torch.no_grad():
      for data in test_loader:
          images, labels = data
          outputs = model(images.to(device))
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels.to(device)).sum().item()

  print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))

def getclassaccuracy(model,test_loader,classes,device):
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
  with torch.no_grad():
      for data in test_loader:
          images, labels = data
          outputs = model(images.to(device))
          _, predicted = torch.max(outputs, 1)
          c = (predicted == labels.to(device)).squeeze()
          for i in range(4):
              label = labels[i]
              class_correct[label] += c[i].item()
              class_total[label] += 1


  for i in range(10):
      print('Accuracy of %5s : %2d %%' % (
          classes[i], 100 * class_correct[i] / class_total[i]))