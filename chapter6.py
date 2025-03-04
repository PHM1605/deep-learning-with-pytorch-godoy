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
