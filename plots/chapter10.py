import numpy as np
import torch 
from matplotlib import pyplot as plt 

# encoding: [N,L,D]=[4,1,256]
def hist_encoding(encoding):
    encoding = encoding.cpu().detach().numpy()
    fig, axs = plt.subplots(1, 4, figsize=(15,4))
    axs = axs.flatten()
    for i in range(4):
        data_point = encoding[i][0] # [256]
        axs[i].hist(data_point, bins=np.linspace(-3,3,15), alpha=0.5)
        axs[i].set_xlabel(f'Data Point #{i}')
        axs[i].set_ylabel(f'# of features')
        axs[i].set_title(f'mean={data_point.mean():.4f}\n var={data_point.var():.4f}', fontsize=16)
        axs[i].set_ylim([0,10])
        axs[i].label_outer()
    fig.tight_layout()
    return fig 

def hist_layer_normed(encoding, normed):
    encoding = encoding.cpu().detach().numpy()
    normed = normed.cpu().detach().numpy()
    fig, axs = plt.subplots(1, 4, figsize=(15,4))
    for i in range(4):
        data_point = encoding[i][0]
        normed_point = normed[i][0]
        axs[i].hist(data_point, bins=np.linspace(-3,3,15), alpha=0.5, label="Original")
        axs[i].hist(normed_point, bins=np.linspace(-3,3,15), alpha=0.5, label="Standardized")
        axs[i].set_xlabel(f'Data Point #{i}')
        axs[i].set_ylabel('# of features')
        axs[i].set_title(f'mean={normed.mean():.4f}\n std={normed.std():.4f}', fontsize=16)
        axs[i].legend()
        axs[i].set_ylim([0,80])
        axs[i].label_outer()
    fig.tight_layout()
    return fig 
