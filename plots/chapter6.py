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

def figure2(first_images, first_labels):
    fig, axs = plt.subplots(1, 6, figsize=(12, 4))
    titles = ['Paper', 'Rock', 'Scissors']
    for i in range(6):
        image, label = ToPILImage()(first_images[i]), first_labels[i]
        axs[i].imshow(image)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(titles[label], fontsize=12)
    fig.tight_layout()
    plt.savefig('test.png')
    return fig 

def figure7(p, disb_outputs):
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    plot_dist(ax, disb_outputs, p)
    plt.savefig('test.png')
    fig.tight_layout()
    return fig 

def plot_dist(ax, distrib_outputs, p):
    ax.hist(distrib_outputs, bins=np.linspace(0,20,21))
    ax.set_xlabel('Sum of Adjusted Outputs')
    ax.set_ylabel('# of Scenarios')
    ax.set_title('p={:.2f}'.format(p))
    ax.set_ylim([0, 500])
    mean_value = distrib_outputs.mean()
    ax.plot([mean_value, mean_value], [0, 500], c='r', linestyle='--', label='Mean={:.2f}'.format(mean_value))
    ax.legend()

# ps: set of dropout probabilities
def figure8(ps=(0.1, 0.3, 0.5, 0.9)):
    spaced_points = torch.linspace(0.1, 1.1, 11)
    fig, axs = plt.subplots(1, 4, figsize=(15,4))
    for ax, p in zip(axs.flat, ps):
        torch.manual_seed(17)
        distrib_outputs = torch.tensor([
            F.linear(F.dropout(spaced_points, p=p), weight=torch.ones(11), bias=torch.tensor(0))
            for _ in range(1000)
        ]) # [1000]
        plot_dist(ax, distrib_outputs, p)
        ax.label_outer()
    fig.tight_layout()
    plt.savefig('test.png')
    return fig
    
def figure9(first_images, seed=17, p=.33):
    torch.manual_seed(seed)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(ToPILImage()(first_images[0]))
    axs[0].set_title('Original Image')
    axs[0].grid(False)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].imshow(ToPILImage()(F.dropout(first_images[:1], p=p)[0]))
    axs[1].set_title('Regular Dropout')
    axs[1].grid(False)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[2].imshow(ToPILImage()(F.dropout2d(first_images[:1], p=p)[0]))
    axs[2].set_title('Two-Dimensional Dropout')
    axs[2].grid(False)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    fig.tight_layout()
    plt.savefig('test.png')
    return fig 

def figure11(losses, val_losses, losses_nodrop, val_losses_nodrop):
    fig, axs = plt.subplots(1, 1, figsize=(10,5))
    axs.plot(losses, 'b', label='Training Losses - Dropout')
    axs.plot(val_losses, 'r', label='Validation Losses - Dropout')
    axs.plot(losses_nodrop, 'b--', label="Training Losses - No Dropout")
    axs.plot(val_losses_nodrop, 'r--', label="Validation Losses - No Dropout")
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Regularizing Effect')
    fig.legend(loc='lower left')
    fig.tight_layout()
    plt.savefig('test.png')
    return fig 
