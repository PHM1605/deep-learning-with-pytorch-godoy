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
