import numpy as np
from PIL import Image 
from copy import deepcopy
import torch 
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms.v2 import Compose, ToImage, Normalize, ToPILImage, Resize, ToDtype
from torchvision.datasets import ImageFolder 
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR, CyclicLR, LambdaLR 
# from stepbystep.v2 import StepByStep 
from data_generation.rps import download_rps 
from plots.chapter6 import *

download_rps()
fig = figure1()

# Pilllow image: RGB
temp_transform = Compose([Resize(28), ToImage(), ToDtype(torch.float32, scale=True)])
temp_dataset = ImageFolder(root='rps', transform=temp_transform)
print("X, y of 1 sample: ", temp_dataset[0][0].shape, temp_dataset[0][1])
temp_loader = DataLoader(temp_dataset, batch_size=16)

## Check statistics for one batch
first_images, first_labels = next(iter(temp_loader))
print("Statistics per channel: ", StepByStep.statistics_per_channel(first_images, first_labels))
## Check statistics for all batches
results = StepByStep.loader_apply(temp_loader, StepByStep.statistics_per_channel)
print("Sum of statistics of all batches: ", results)
normalizer = StepByStep.make_normalizer(temp_loader)
print("Normalizer: ", normalizer)

## The real dataset 
composer = Compose([
    Resize(28),
    ToImage(),
    ToDtype(torch.float32, scale=True),
    normalizer
])
train_data = ImageFolder(root='rps', transform=composer)
val_data = ImageFolder(root='rps-test-set', transform=composer)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)
torch.manual_seed(88)
first_images, first_labels = next(iter(train_loader))
fig = figure2(first_images, first_labels)

class CNN2(nn.Module):
    def __init__(self, n_filters, p=0.0):
        super(CNN2, self).__init__()
        self.n_filters = n_filters
        self.p = p
        # Featurizer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_filters, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3)
        # Classifier
        self.fc1 = nn.Linear(n_filters*5*5, 50)
        self.fc2 = nn.Linear(50, 3)
        self.drop = nn.Dropout(self.p)

    def featurizer(self, x):
        x = self.conv1(x) # [3,28,28]->[n_filters,26,26]
        x = F.relu(x)
        x = F.maxpool2d(x, kernel_size=2) # [n_filters,13,13]
        x = self.conv2(x) # [n_filters,11,11]
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # [n_filters,5,5]
        x = nn.Flatten()(x)
        return x 
    
    def classifier(self, x):
        if self.p > 0:
            x = self.drop(x) # [n_filters*5*5]
        x = self.fc1(x) # [50]
        x = F.relu(x)
        if self.p > 0:
            x = self.drop(x)
        x = self.fc2(x)
        return x 
    
    def forward(self, x):
        x = self.featurizer(x)
        x = self.classifier(x)
        return x 
    
## Dropout
dropping_model = nn.Sequential(nn.Dropout(p=0.5))
spaced_points = torch.linspace(0.1, 1.1, 11)
print("Input spaced points: ", spaced_points)
torch.manual_seed(44)
dropping_model.train()
output_train = dropping_model(spaced_points)
# Some outputs are demolished to 0; the rest are scaled by 1/p (in this case, multiplying by 2)
print("Output of dropping-model: ", output_train)
print("Weight of adjusted output: ", F.linear(output_train, weight=torch.ones(11), bias=torch.tensor(0))) # 9.4
# Why need adjusting? Because there is no dropout in 'eval' mode
dropping_model.eval()
output_eval = dropping_model(spaced_points)
print("Output of non-dropping (eval) model: ", output_eval)
print("Weight of non-adjusted eval output: ", F.linear(output_eval, weight=torch.ones(11), bias=torch.tensor(0))) # 6.6
# Run the dropout experiments 1000 times, take the value of adjusted dropout outputs -> calculate their weight-sum, we will see the mean of those 1000 samples is closed to 6.6
torch.manual_seed(17)
p = 0.5
distrib_outputs = torch.tensor([
    F.linear(F.dropout(spaced_points, p=p), weight=torch.ones(11), bias=torch.tensor(0))
    for _ in range(1000)
])
fig = figure7(p, distrib_outputs)
fig = figure8()
