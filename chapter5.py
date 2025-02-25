import random
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, Normalize
from data_generation.image_classification import generate_dataset
# from helpers import index_splitter, make_balanced_sampler
# from stepbystep.v1 import StepByStep 

## Convolution
# Simple arrays illustration of convolution
single = np.array(
    [[
        [[5,0,8,7,8,1],
         [1,9,5,0,7,7],
         [6,0,2,4,6,6],
         [9,7,6,6,8,4],
         [8,3,8,5,1,3],
         [7,2,7,0,1,0]]
    ]]
)
print("Single shape: ", single.shape) # [1,1,6,6]
identity = np.array([[
    [[0,0,0],
     [0,1,0],
     [0,0,0]]
]])
print("Identity shape: ", identity.shape)
region = single[:, :, 0:3, 0:3]
filtered_region = region * identity
print("First filted region: ", filtered_region)
# Moving around the filter
new_region = single[:, :, 0:3, (0+3):(3+3)]
new_filtered_region = new_region * identity
print("Next filtered region: ", new_filtered_region)

## NOTE: Shape after a convolution: (hi,wi)*(hf,wf) = ((hi+2*padding-hf+1)/stride, (wi+2*padding-wf+1)/stride)

## Convolution with Pytorch
image = torch.as_tensor(single).float()
kernel_identity = torch.as_tensor(identity).float()
convolved = F.conv2d(image, kernel_identity, stride=1)
print("Convolved with torch with functional: ", convolved)
conv = nn.Conv2d(
    in_channels=1, out_channels=1, kernel_size=3, stride=1
)
print("Convolved with torch with module (1 filter): ", conv(image))
conv_multiple = nn.Conv2d(
    in_channels=1, out_channels=2, kernel_size=3, stride=1
)
print("Convolved with torch with module (2 filters): ", conv_multiple(image))
# To set a convolution layer with specific chosen weights
print("TEST: ", conv.weight[0].shape, kernel_identity.shape)
with torch.no_grad():
    conv.weight[0] = kernel_identity
    conv.bias[0] = 0
print("Convolution after setting filter weights to identity: ", conv(image))
# Convolution with stride = 2
convolved_stride2 = F.conv2d(image, kernel_identity, stride=2)
print("Convolution with stride = 2: ", convolved_stride2)

## Padding
constant_padder = nn.ConstantPad2d(padding=1, value=0)
print("Padded with constant image: ", constant_padder(image))
padded = F.pad(image, pad=(1,1,1,1), mode='constant', value=0) # pad in directions [left,right,top,bottom]
print("Padded with functional directions: ", padded)
# Replication padding with F.pad(mode='replicate') or below:
replication_padder = nn.ReplicationPad2d(padding=1)
print("Replication padded image: ", replication_padder(image))
# Reflection padding with F.pad(mode='reflect') or below:
reflection_padder = nn.ReflectionPad2d(padding=1)
print("Reflection padded image: ", reflection_padder(image))
# Circular padding with F.pad(mode='circular') (no module available)
print("Circular padding: ", F.pad(image, pad=(1,1,1,1), mode='circular'))

## Edge filter
edge = np.array([[
    [[0,1,0],
     [1,-4,1],
     [0,1,0]]
    ]])
kernel_edge = torch.as_tensor(edge).float()
print("Edge kernel shape: ", kernel_edge.shape)

