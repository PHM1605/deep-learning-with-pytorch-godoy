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

# ## RNN Layer
# n_features = 2
# hidden_dim = 2
# torch.manual_seed(19)
# rnn = nn.RNN(input_size=n_features, hidden_size=hidden_dim)
# print("RNN layer state dict: ", rnn.state_dict())

# ## Checking a batch of three sequences, each includes 4 corners
# batch = torch.as_tensor(np.array(points[:3])).float()
# print("Batch shape: ", batch.shape) # [3,4,2]
# torch.manual_seed(19)
# rnn_batch_first = nn.RNN(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
# out, final_hidden = rnn_batch_first(batch)
# # NOTICE: we get "batch-fist-output" and "sequence-first-hidden"
# print("Output shape and final hidden shape: ", out.shape, final_hidden.shape) # [3,4,2] and [1,3,2]

# ## Stacked RNN
# torch.manual_seed(19)
# rnn_stacked = nn.RNN(input_size=2, hidden_size=2, num_layers=2, batch_first=True)
# state = rnn_stacked.state_dict()
# print("Stacked RNN state: ", rnn_stacked.state_dict())
# # Stacked RNN manually
# rnn_layer0 = nn.RNN(input_size=2, hidden_size=2, batch_first=True)
# rnn_layer1 = nn.RNN(input_size=2, hidden_size=2, batch_first=True)
# # applying 'dict' to [('a',1),('b',2)] => {'a':1, 'b':2}
# rnn_layer0.load_state_dict(dict(list(state.items())[:4]))
# rnn_layer1.load_state_dict(dict(
#     [ (k[:-1]+'0', v) 
#         for k,v in list(state.items())[4:]
#     ]
# ))
# x = torch.as_tensor(points[0:1]).float() # [1,4,2]
# out0, h0 = rnn_layer0(x)
# out1, h1 = rnn_layer1(out0)
# # Overall output of stacked RNN includes: hidden-states-sequence + concatenation of hidden states of all layers
# print("Output of the stacked RNN: ", out1, torch.cat([h0, h1]))
# out, hidden = rnn_stacked(x)
# print("Comparing with built-in library: ", out, hidden)
# # NOTICE: don't forget that 'out' is batch-first, 'hidden' is layers-first
# print("Last element of output is hidden state of last layer? ", (out[:,-1] == hidden.permute(1,0,2)[:,-1]).all())

# ## Bidirectional RNN 
# torch.manual_seed(19)
# rnn_bidirect = nn.RNN(input_size=2, hidden_size=2, bidirectional=True, batch_first=True)
# state = rnn_bidirect.state_dict()
# print("Bidirectional RNN state:\n ", state)
# rnn_forward = nn.RNN(input_size=2, hidden_size=2, batch_first=True)
# rnn_reverse = nn.RNN(input_size=2, hidden_size=2, batch_first=True)
# rnn_forward.load_state_dict(dict(list(state.items())[:4]))
# rnn_reverse.load_state_dict(dict([
#     (k[:-8], v)
#     for k, v in list(state.items())[4:]
# ]))
# x_rev = torch.flip(x, dims=[1]) # [1,4,2]
# out, h = rnn_forward(x) # [1,4,2], [1,1,2]
# print(out.shape, h.shape)
# out_rev, h_rev = rnn_reverse(x_rev) # [1,4,2], [1,1,2]
# out_rev_back = torch.flip(out_rev, dims=[1]) # [1,4,2]
# print("Total output and total final hidden state: ", 
#     torch.cat([out, out_rev_back], dim=2), # [1,4,4]
#     torch.cat([h, h_rev])) # [1,2,2]
# # check with built-in bidirectional RNN
# out, hidden = rnn_bidirect(x) # [1,4,4], [1,2,2]
# print("Total output and total final hidden state (built in): ", out, hidden)
# print("Is last output element same as final hidden state? ", out[:,-1] == hidden.permute(1,0,2).view(1,-1)) # No

## Square model
test_points, test_directions = generate_sequences(seed=19)
train_data = TensorDataset(
    torch.as_tensor(points).float(),
    torch.as_tensor(directions).view(-1,1).float()
)
test_data = TensorDataset(
    torch.as_tensor(test_points).float(),
    torch.as_tensor(test_directions).view(-1,1).float()
)
train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True
)
test_loader = DataLoader(
    test_data, batch_size=16
)

class SquareModel(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs):
        super(SquareModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features 
        self.n_outputs = n_outputs 
        self.hidden = None 
        self.basic_rnn = nn.RNN(self.n_features, self.hidden_dim, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)
    def forward(self, X):
        # X: [N, L, F]
        # output: [N, L, H]
        # final hidden state: [1, N, H]
        batch_first_output, self.hidden = self.basic_rnn(X)
        last_output = batch_first_output[:, -1] # [N,1,H]
        out = self.classifier(last_output) # [N,1,n_outputs]
        return out.view(-1, self.n_outputs)

torch.manual_seed(21)
model = SquareModel(n_features=2, hidden_dim=2, n_outputs=1)
loss = nn.BCEWithLogitsLoss()
optimizer= optim.Adam(model.parameters(), lr=0.01)
sbs_rnn = StepByStep(model, loss, optimizer)
sbs_rnn.set_loaders(train_loader, test_loader)
sbs_rnn.train(100)
fig = sbs_rnn.plot_losses()
print("Recall on test dataset: ", StepByStep.loader_apply(test_loader, sbs_rnn.correct)) # 99%

## Visualizing the model
# Transformed input 
state = model.basic_rnn.state_dict()
print("Square model state on input: ", state['weight_ih_l0'], state['bias_ih_l0'])
fig = figure13(model.basic_rnn)