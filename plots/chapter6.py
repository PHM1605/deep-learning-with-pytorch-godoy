import numpy as np 
import torch 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import pandas as pd
from copy import deepcopy
from PIL import Image 
from stepbystep.v2 import StepByStep 
from torchvision.transforms import ToPILImage 
from sklearn.linear_model import LinearRegression 
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR, CyclicLR, LambdaLR


def figure1(folder = 'rps'):
    pass