import numpy as np
from PIL import Image 
import torch
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset, random_split 
from torchvision.transforms.v2 import Compose, ToImage, Normalize, Resize, ToPILImage, CenterCrop, RandomResizedCrop, ToDtype 
from torchvision.datasets import ImageFolder 
from torchvision.models import alexnet, resnet18, inception_v3 
from torchvision.models.alexnet import AlexNet_Weights
from torch.hub import load_state_dict_from_url 
from stepbystep.v3 import StepByStep 
from plots.chapter7 import *

## Comparing Architectures
fig = figure1()

alex = alexnet(weights=None)
print("AlexNet structure: ", alex)