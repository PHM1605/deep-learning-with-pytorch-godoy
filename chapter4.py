import random
import numpy as np 
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler, SubsetRandomSampler 
from torchvision.transforms.v2 import Compose, ToImage, Normalize, ToPILImage, RandomHorizontalFlip, Resize, ToDtype
import matplotlib.pyplot as plt
from data_generation.image_classification import generate_dataset 
from stepbystep.v0 import StepByStep
from plots.chapter4 import *

images, labels = generate_dataset(img_size=5, n_images=300, binary=True, seed=13)
fig = plot_images(images, labels, n_plot=30)

## Images and Channels
image_r = np.zeros((5, 5), dtype=np.uint8)
image_r[:, 0] = 255
image_r[:, 1] = 128
image_g = np.zeros((5, 5), dtype=np.uint8)
image_g[:, 1] = 128
image_g[:, 2] = 255
image_g[:, 3] = 128
image_b = np.zeros((5, 5), dtype=np.uint8)
image_b[:, 3] = 128
image_b[:, 4] = 255
# This formula is to convert RGB image to gray scale image
image_gray = 0.2126 * image_r + 0.7152 * image_g + 0.0722 * image_b 
image_rgb = np.stack([image_r, image_g, image_b], axis=2)
fig = image_channels(image_r, image_g, image_b, image_rgb, image_gray, rows=(0, 1))
fig = image_channels(image_r, image_g, image_b, image_rgb, image_gray, rows=(0, 2))

## Shape: NCHW (Pytorch) vs NHWC (Tensorflow)
print("Images shape: ", images.shape) # [300,1,5,5]
example = images[7] # [1,5,5]
example_hwc = np.transpose(example, (1,2,0)) # [5,5,1]
image_tensor = ToImage()(example_hwc) # convert to tensor, integer and [0,255]
print("Image tensor and shape: ", image_tensor, image_tensor.shape)
print("Is Image a tensor? ", isinstance(image_tensor, torch.Tensor))
example_tensor = ToDtype(torch.float32, scale=True)(image_tensor) # convert from "int [0,255]" to "float [0,1]"
print("Scaled float tensor: ", example_tensor)

## Shorten the above steps
def ToTensor():
    return Compose([ToImage(), ToDtype(torch.float32, scale=True)])
tensorizer = ToTensor()
example_tensor = tensorizer(example_hwc)
print("Scaled float tensor: ", example_tensor)
example_img = ToPILImage()(example_tensor) # convert tensor to PIL Image
print("Type of PIL image: ", type(example_img))
plt.imshow(example_img, cmap='gray')
plt.grid(False)

