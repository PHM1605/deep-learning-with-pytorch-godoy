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
    paper = Image.open(f'{folder}/paper/paper02-089.png')
    rock = Image.open(f'{folder}/rock/rock06ck02-100.png')
    scissors = Image.open(f'{folder}/scissors/testscissors02-006.png')
    images = [rock, paper, scissors]
    titles = ['Rock', 'Paper', 'Scissors']
    fig, axs = plt.subplots(1, 3, figsize=(12,5))
    for ax, image, title in zip(axs, images, titles):
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    plt.savefig('test.png')
    return fig 
