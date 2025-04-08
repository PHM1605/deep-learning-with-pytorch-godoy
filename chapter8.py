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

# ## RNN Cell
# hidden_state = torch.zeros(2)
# n_features = 2
# hidden_dim = 2
# torch.manual_seed(19)
# rnn_cell = nn.RNNCell(input_size=n_features, hidden_size=hidden_dim)
# rnn_state = rnn_cell.state_dict()
# print("Current RNN Cell state: ", rnn_state)
# # RNN one-cell updating manually
# linear_input = nn.Linear(n_features, hidden_dim)
# linear_hidden = nn.Linear(hidden_dim, hidden_dim)
# # how to set weight of a layer manually
# with torch.no_grad():
#     linear_input.weight = nn.Parameter(rnn_state['weight_ih'])
#     linear_input.bias = nn.Parameter(rnn_state['bias_ih'])
#     linear_hidden.weight = nn.Parameter(rnn_state['weight_hh'])
#     linear_hidden.bias = nn.Parameter(rnn_state['bias_hh'])
# initial_hidden = torch.zeros(1, hidden_dim)
# th = linear_hidden(initial_hidden) # transformed hidden state [1,2]
# X = torch.as_tensor(points[0]).float() # [4,2] - four corners
# print("First sequence: ", X)
# tx = linear_input(X[0:1]) # transformed input state [1,2]
# adding = th + tx
# print("RNN updated hidden state manually: ", torch.tanh(adding))
# print("RNN updated hidden state with torch library: ", rnn_cell(X[0:1]))
# fig = figure8(linear_hidden, linear_input, X)
# # Running through all 4 corners
# hidden = torch.zeros(1, hidden_dim)
# for i in range(X.shape[0]):
#     out = rnn_cell(X[i:i+1], hidden)
#     print("Next hidden state: ", out)
#     hidden = out 

## RNN Layer
n_features = 2
hidden_dim = 2
torch.manual_seed(19)
rnn = nn.RNN(input_size=n_features, hidden_size=hidden_dim)
print("RNN layer state dict: ", rnn.state_dict())

# ## Checking a batch of three sequences, each includes 4 corners
# batch = torch.as_tensor(np.array(points[:3])).float()
# print("Batch shape: ", batch.shape) # [3,4,2]
# torch.manual_seed(19)
# rnn_batch_first = nn.RNN(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
# out, final_hidden = rnn_batch_first(batch)
# # NOTICE: we get "batch-fist-output" and "sequence-first-hidden"
# print("Output shape and final hidden shape: ", out.shape, final_hidden.shape) # [3,4,2] and [1,3,2]

## Stacked RNN
torch.manual_seed(19)
rnn_stacked = nn.RNN(input_size=2, hidden_size=2, num_layers=2, batch_first=True)
state = rnn_stacked.state_dict()
print("Stacked RNN state: ", rnn_stacked.state_dict())
# Stacked RNN manually
rnn_layer0 = nn.RNN(input_size=2, hidden_size=2, batch_first=True)
rnn_layer1 = nn.RNN(input_size=2, hidden_size=2, batch_first=True)
# applying 'dict' to [('a',1),('b',2)] => {'a':1, 'b':2}
rnn_layer0.load_state_dict(dict(list(state.items())[:4]))
rnn_layer1.load_state_dict(dict(
    [ (k[:-1]+'0', v) 
        for k,v in list(state.items())[4:]
    ]
))
x = torch.as_tensor(points[0:1]).float() # [1,4,2]
out0, h0 = rnn_layer0(x)
out1, h1 = rnn_layer1(out0)
# Overall output of stacked RNN includes: hidden-states-sequence + concatenation of hidden states of all layers
print("Output of the stacked RNN: ", out1, torch.cat([h0, h1]))
out, hidden = rnn_stacked(x)
print("Comparing with built-in library: ", out, hidden)
# NOTICE: don't forget that 'out' is batch-first, 'hidden' is sequence-first
print("Last element of output is hidden state of last layer? ", (out[:,-1] == hidden.permute(1,0,2)[:,-1]).all())