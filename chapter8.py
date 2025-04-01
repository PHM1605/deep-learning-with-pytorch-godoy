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

## RNN
hidden_state = torch.zeros(2)
n_features = 2
hidden_dim = 2
torch.manual_seed(19)
rnn_cell = nn.RNNCell(input_size=n_features, hidden_size=hidden_dim)
rnn_state = rnn_cell.state_dict()
print("Current RNN Cell state: ", rnn_state)
# RNN one-cell updating manually
linear_input = nn.Linear(n_features, hidden_dim)
linear_hidden = nn.Linear(hidden_dim, hidden_dim)
# how to set weight of a layer manually
with torch.no_grad():
    linear_input.weight = nn.Parameter(rnn_state['weight_ih'])
    linear_input.bias = nn.Parameter(rnn_state['bias_ih'])
    linear_hidden.weight = nn.Parameter(rnn_state['weight_hh'])
    linear_hidden.bias = nn.Parameter(rnn_state['bias_hh'])
initial_hidden = torch.zeros(1, hidden_dim)
th = linear_hidden(initial_hidden) # transformed hidden state [1,2]
X = torch.as_tensor(points[0]).float() # [4,2] - four corners
print("First sequence: ", X)
tx = linear_input(X[0:1]) # transformed input state [1,2]
adding = th + tx
print("RNN updated hidden state manually: ", torch.tanh(adding))
print("RNN updated hidden state with torch library: ", rnn_cell(X[0:1]))
fig = figure8(linear_hidden, linear_input, X)