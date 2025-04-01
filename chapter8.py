import numpy as np
import torch
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset 
from torch.nn.utils import rnn as rnn_utils 
from data_generation.square_sequences import generate_sequences 
from stepbystep.v4 import StepByStep
from plots.chapter8 import *

# # plot 4 points without direction
# fig = counter_vs_clock(draw_arrows=False)

# # plot counter-clockwise and clockwise
# fig = counter_vs_clock()

# # Possible sequences of corners and corresponding outputs
# fig = plot_sequences()

points, directions = generate_sequences(n=128, seed=13)
fig = plot_data(points, directions)