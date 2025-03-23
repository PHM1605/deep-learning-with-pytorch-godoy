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

def EWMA(past_value, current_value, alpha):
    return (1-alpha) * past_value + alpha*current_value 

def calc_ewma(values, period):
    alpha = 2/(period+1)
    result = []
    for v in values:
        try: 
            prev_value = result[-1]
        except IndexError:
            prev_value = 0
        new_value = EWMA(prev_value, v, alpha)
        result.append(new_value)
    return np.array(result)

# for bias-corrected EWMA (for the first few samples)
def correction(averaged_value, beta, steps):
    return averaged_value / (1 - beta**steps)

def calc_corrected_ewma(values, period):
    ewma = calc_ewma(values, period)
    alpha = 2/(period+1)
    beta = 1 - alpha 
    result = []
    for step, v in enumerate(ewma):
        adj_value = correction(v, beta, step+1)
        result.append(adj_value)
    return np.array(result)

def ma_vs_ewma(values, periods=19):
    # min_periods: minimum number of observations in window required to have a value
    # window: size of moving window
    ma19 = pd.Series(values).rolling(min_periods=0, window=periods).mean()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(values, c='k', label='Temperatures')
    ax.plot(ma19, c='k', linestyle='--', label='MA')
    ax.plot(calc_ewma(values, periods), c='r', linestyle='--', label='EWMA')
    ax.plot(calc_corrected_ewma(values, periods), c='r', linestyle='-', label='Bias-corrected EWMA')
    ax.set_title('MA vs EWMA')
    ax.set_ylabel('Temperature')
    ax.set_xlabel('Days')
    ax.legend(fontsize=12)
    fig.tight_layout()
    plt.savefig('test.png')
    return fig 

def compare_optimizers(model, loss_fn, optimizers, train_loader, val_loader=None, schedulers=None, layers_to_hook='', n_epochs=50):
    from stepbystep.v3 import StepByStep
    results = {}
    model_state = deepcopy(model).state_dict()
    # optimizers: {'SGD': {'class': optim.SGD, 'parms': {'lr': 0.1}}, 'Adam': {...}}
    for desc, opt in optimizers.items():
        model.load_state_dict(model_state)
        optimizer = opt['class'](model.parameters(), **opt['parms'])
        sbs = StepByStep(model, loss_fn, optimizer)
        sbs.set_loaders(train_loader, val_loader)

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

def figure15(alpha=1/3, periods=5, steps=10):
    t = np.arange(1, steps+1)
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.bar(t-1, alpha*(1-alpha)**(t-1), label='EWMA')
    ax.bar(t-1, [1/periods]*periods+[0]*(10-periods), color='r', alpha=0.3, label='MA')
    ax.set_xticks(t-1)
    ax.grid(False)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Weight")
    ax.set_title(r'$EWMA\ \alpha=\frac{1}{3}$ vs MA (5 periods)')
    ax.legend()
    fig.tight_layout()
    plt.savefig("test.png")
    return fig

def figure17(gradients, corrected_gradients, corrected_sq_gradients, adapted_gradients):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax = axs[0]
    ax.plot(gradients, c='k', label=r'Gradients')
    ax.plot(corrected_gradients, c='r', linestyle='-', label=r'$Bias-corrected\ EWMA(grad)$')
    ax.set_title('EWMA for Smoothing')
    ax.set_ylabel('Gradient')
    ax.set_xlabel('Mini-batches')
    ax.set_ylim([-1.5, 1.5])
    ax.legend(fontsize=12)

    ax = axs[1]
    ax.plot(1/(np.sqrt(corrected_sq_gradients)+1e-8), c='b', linestyle='-', label=r'$\frac{1}{\sqrt{Bias-corrected\ EWMA(grad^2)}}$')
    ax.set_title('EWMA for Scaling')
    ax.set_ylabel('Factor')
    ax.set_xlabel('Mini-batches')
    ax.set_ylim([0, 5])
    ax.legend(fontsize=12)

    ax = axs[2]
    ax.plot(gradients, c='k', label='Gradients')
    ax.plot(adapted_gradients, c='g', label='Adapted Gradients')
    ax.set_title('Gradients')
    ax.set_ylabel('Gradient')
    ax.set_xlabel('Mini-batches')
    ax.set_ylim([-1.5, 1.5])
    ax.legend(fontsize=12)

    fig.tight_layout()
    plt.savefig('test.png')
    return fig
