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

# points, directions = generate_sequences(n=128, seed=13)
# fig = plot_data(points, directions)

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

# ## Data preparation
# test_points, test_directions = generate_sequences(seed=19)
# train_data = TensorDataset(
#     torch.as_tensor(np.array(points)).float(),
#     torch.as_tensor(directions).view(-1,1).float()
# )
# test_data = TensorDataset(
#     torch.as_tensor(np.array(test_points)).float(),
#     torch.as_tensor(test_directions).view(-1,1).float()
# )
# train_loader = DataLoader(
#     train_data, batch_size=16, shuffle=True
# )
# test_loader = DataLoader(
#     test_data, batch_size=16
# )

# # Square model
# class SquareModel(nn.Module):
#     def __init__(self, n_features, hidden_dim, n_outputs):
#         super(SquareModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.n_features = n_features 
#         self.n_outputs = n_outputs 
#         self.hidden = None 
#         self.basic_rnn = nn.RNN(self.n_features, self.hidden_dim, batch_first=True)
#         self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)
#     def forward(self, X):
#         # X: [N, L, F]
#         # output: [N, L, H]
#         # final hidden state: [1, N, H]
#         batch_first_output, self.hidden = self.basic_rnn(X)
#         last_output = batch_first_output[:, -1] # [N,1,H]
#         out = self.classifier(last_output) # [N,1,n_outputs]
#         return out.view(-1, self.n_outputs)

# torch.manual_seed(21)
# model = SquareModel(n_features=2, hidden_dim=2, n_outputs=1)
# loss = nn.BCEWithLogitsLoss()
# optimizer= optim.Adam(model.parameters(), lr=0.01)
# sbs_rnn = StepByStep(model, loss, optimizer)
# sbs_rnn.set_loaders(train_loader, test_loader)
# sbs_rnn.train(100)
# fig = sbs_rnn.plot_losses()
# print("Recall on test dataset: ", StepByStep.loader_apply(test_loader, sbs_rnn.correct)) # 99%

# ## Visualizing the model
# # Transformed input of 4 basic corners
# state = model.basic_rnn.state_dict()
# print("Square model state on input: ", state['weight_ih_l0'], state['bias_ih_l0'])
# fig = figure13(model.basic_rnn)
# # Final hidden states on 4 toying clockwise- and 4 counterclockwise-combinations-of-corners (totally 8)
# fig = canonical_contour(model)

# # Final hidden states of our real sequence (128 samples)
# fig = hidden_states_contour(model, points, directions) # 128 samples, each [4,2] is a sequence to be predicted

# ## Hidden state observation after EACH OPERATION (column) when EACH CORNER (row) arrives
# # last column of one row is the input (1st column) of the next row
# fig = figure16(model.basic_rnn)
# # hidden point progression after EACH POINT (each different color) arrives
# fig = figure17(model.basic_rnn)

# ## GRU - Gated Recurrent Unit 
# # r=ResetGate, z=UpdateGate; [1,1] and [0,0] respectively for normal RNN cell
# # h_new = (1-z)*tanh(r*t_h + t_x) + z*h_old 
# # r and z are learnt from two RNN cells with softmax-activation
# n_features = 2
# hidden_dim = 2
# torch.manual_seed(17)
# gru_cell = nn.GRUCell(input_size=n_features, hidden_size=hidden_dim)
# gru_state = gru_cell.state_dict()
# print("GRU State: ", gru_state) # GRU cell concatenates 3 shapes for r, z and n (main RNN cell)
# Wx, bx = gru_state['weight_ih'], gru_state['bias_ih'] # [6,2], [6], #columns = #input features
# Wh, bh = gru_state['weight_hh'], gru_state['bias_hh'] # [6,2], [6]
# Wxr, Wxz, Wxn = Wx.split(hidden_dim, dim=0) # each [2,2]
# bxr, bxz, bxn = bx.split(hidden_dim, dim=0) # each [2]
# Whr, Whz, Whn = Wh.split(hidden_dim, dim=0) # each [2,2]
# bhr, bhz, bhn = bh.split(hidden_dim, dim=0) # each [2]

def linear_layers(Wx, bx, Wh, bh):
    hidden_dim, n_features = Wx.size()
    lin_input = nn.Linear(n_features, hidden_dim)
    lin_input.load_state_dict({'weight':Wx, 'bias':bx})
    lin_hidden = nn.Linear(hidden_dim, hidden_dim)
    lin_hidden.load_state_dict({'weight':Wh, 'bias':bh})
    return lin_hidden, lin_input 

# # reset gate layers - red
# r_hidden, r_input = linear_layers(Wxr, bxr, Whr, bhr)
# # update gate layers - blue 
# z_hidden, z_input = linear_layers(Wxz, bxz, Whz, bhz)
# # candidate state layers - black
# n_hidden, n_input = linear_layers(Wxn, bxn, Whn, bhn)

# def reset_gate(h, x):
#     thr = r_hidden(h)
#     txr = r_input(x)
#     r = torch.sigmoid(thr+txr)
#     return r 

# def update_gate(h, x):
#     thz = z_hidden(h)
#     txz = z_input(x)
#     z = torch.sigmoid(thz + txz)
#     return z 

# def candidate_n(h, x, r):
#     thn = n_hidden(h)
#     txn = n_input(x)
#     n = torch.tanh(r*thn+txn)
#     return n 

# initial_hidden = torch.zeros(1, hidden_dim)
# X = torch.as_tensor(points[0]).float()
# first_corner = X[0:1]
# r = reset_gate(initial_hidden, first_corner)
# print('r: ', r) # [1,2]
# n = candidate_n(initial_hidden, first_corner, r)
# print('n: ', n)
# z = update_gate(initial_hidden, first_corner)
# print('z: ', z)
# h_prime = (1-z)*n + z*initial_hidden 
# print('h_prime: ', h_prime)
# print('Comparing with built-in lib: ', gru_cell(first_corner))

# class SquareModelGRU(nn.Module):
#     def __init__(self, n_features, hidden_dim, n_outputs):
#         super(SquareModelGRU, self).__init__()
#         self.hidden_dim = hidden_dim 
#         self.n_features = n_features 
#         self.n_outputs = n_outputs 
#         self.hidden = None
#         self.basic_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True)
#         self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)
    
#     def forward(self, X):
#         # X is batch-first: [N,L,F]; output is batch-first: [N,L,H]
#         # final hidden state is batch-second: [1,N,H]
#         batch_first_output, self.hidden = self.basic_rnn(X)
#         last_output = batch_first_output[:,-1] # [N,1,H]
#         out = self.classifier(last_output)
#         return out.view(-1, self.n_outputs) # [N,n_outputs]

# torch.manual_seed(21)
# model = SquareModelGRU(n_features=2, hidden_dim=2, n_outputs=1)
# loss = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# sbs_gru = StepByStep(model, loss, optimizer)
# sbs_gru.set_loaders(train_loader, test_loader)
# sbs_gru.train(100)
# fig = sbs_gru.plot_losses()
# print("Validation result: ", StepByStep.loader_apply(test_loader, sbs_gru.correct)) # recall: 100%

# ## Visualizing the model
# # hidden states of RNN and GRU for 1 clockwise and 1 counter-clockwise sequence 
# fig = figure20(sbs_rnn.model, sbs_gru.model)
# plt.savefig('test.png')
# # hidden states of RNN and GRU for all sequences
# fig = hidden_states_contour(model, points, directions)
# plt.savefig('test.png')
# # hidden state over every operation performed inside the GRU (for 1 sequence of 4 corners)
# fig = figure22(model.basic_rnn)

# ## LSTM 
# # candidate hidden state (short-term memory): g = tanh(thg + txg)
# # new cell state (long-term-memory): c' = i*g + f*c; with i=input gate, f=forget gate; c=old cell state, g=candidate hidden state
# # new hidden state: h' = o*tanh(c'); with o=output gate
# # all gates have the form: i=sigmoid(thi+txi) with th = W*h+bias, tx=W*x+bias
# n_features = 2
# hidden_dim = 2
# torch.manual_seed(17)
# lstm_cell = nn.LSTMCell(input_size=n_features, hidden_size=hidden_dim)
# lstm_state = lstm_cell.state_dict()
# print("LSTM state: ", lstm_state) # weight_ih:[8,2], weight_hh:[8,2], bias_ih:[8], bias_hh:[8]
# Wx, bx = lstm_state['weight_ih'], lstm_state['bias_ih']
# Wh, bh = lstm_state['weight_hh'], lstm_state['bias_hh']
# Wxi, Wxf, Wxg, Wxo = Wx.split(hidden_dim, dim=0)
# bxi, bxf, bxg, bxo = bx.split(hidden_dim, dim=0)
# Whi, Whf, Whg, Who = Wh.split(hidden_dim, dim=0)
# bhi, bhf, bhg, bho = bh.split(hidden_dim, dim=0)

# i_hidden, i_input = linear_layers(Wxi, bxi, Whi, bhi)
# f_hidden, f_input = linear_layers(Wxf, bxf, Whf, bhf)
# o_hidden, o_input = linear_layers(Wxo, bxo, Who, bho)

# g_cell = nn.RNNCell(n_features, hidden_dim)
# g_cell.load_state_dict({
#     'weight_ih': Wxg, 'bias_ih': bxg,
#     'weight_hh': Whg, 'bias_hh': bhg
# })

def forget_gate(h, x):
    thf = f_hidden(h)
    txf = f_input(x)
    f = torch.sigmoid(thf + txf)
    return f 

def output_gate(h, x):
    tho = o_hidden(h)
    txo = o_input(x)
    o = torch.sigmoid(tho + txo)
    return o 

def input_gate(h, x):
    thi = i_hidden(h)
    txi = i_input(x)
    i = torch.sigmoid(thi + txi)
    return i 

# initial_hidden = torch.zeros(1, hidden_dim)
# initial_cell = torch.zeros(1, hidden_dim)
# X = torch.as_tensor(points[0]).float() # [4,2]
# first_corner = X[0:1] # [1,2]
# g = g_cell(first_corner) #
# i = input_gate(initial_hidden, first_corner)
# gated_input = g * i # [1,2]
# f = forget_gate(initial_hidden, first_corner)
# gated_cell = initial_cell * f # [[0,0]]
# c_prime = gated_cell + gated_input 
# o = output_gate(initial_hidden, first_corner)
# h_prime = o * torch.tanh(c_prime)
# print("LSTM cell returning state: ", h_prime, c_prime)
# print("Comparing with built-in library LSTM: ", lstm_cell(first_corner))

class SquareModelLSTM(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs):
        super(SquareModelLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features 
        self.n_outputs = n_outputs 

        self.hidden = None # final hidden state
        self.cell = None # final cell state 
        self.basic_rnn = nn.LSTM(self.n_features, self.hidden_dim, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)
    
    def forward(self, X):
        # X: [N,L,F]; batch_first_output: [N,L,H]; final hidden state: [1,N,H], final cell state:[1,N,H]
        batch_first_output, (self.hidden, self.cell) = self.basic_rnn(X)
        last_output = batch_first_output[:,-1] # [N,1,H]
        out = self.classifier(last_output) # [N,1,n_outputs]
        return out.view(-1, self.n_outputs)

# torch.manual_seed(21)
# model = SquareModelLSTM(n_features=2, hidden_dim=2, n_outputs=1)
# loss = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# sbs_lstm = StepByStep(model, loss, optimizer)
# sbs_lstm.set_loaders(train_loader, test_loader)
# sbs_lstm.train(100)
# fig = sbs_lstm.plot_losses()
# print("Square LSTM model validation result: ", StepByStep.loader_apply(test_loader, sbs_lstm.correct))

# # Visualizing the hidden state
# # for 1 clockwise and 1 counter-clockwise sequence
# fig = figure25(sbs_rnn.model, sbs_gru.model, sbs_lstm.model)
# plt.savefig('test.png')
# # for all sequences 
# fig = hidden_states_contour(model, points, directions)
# plt.savefig('test.png')

# ## Variable length sequences
# s0 = points[0] # 4 data points
# s1 = points[1][2:] # 2 data points 
# s2 = points[2][1:] # 3 data points 
# all_seqs = [s0, s1, s2]

# # Zero padding to ensure all have same size
# seq_tensors = [torch.as_tensor(seq).float() for seq in all_seqs]
# padded = rnn_utils.pad_sequence(seq_tensors, batch_first=True)

# torch.manual_seed(11)
# rnn = nn.RNN(2, 2, batch_first=True)
# output_padded, hidden_padded = rnn(padded) # [3,4,2], [1,3,2] (3 sequences, 4 corners, dimension of 2 for each corner)
# print("Output of padded input: ", output_padded)
# print("Final hidden of padded input: ", hidden_padded.permute(1,0,2)) # to convert to [3,1,2]

# # Packing instead of zero-padding
# # enfore_sorted: requiring seq_tensors to be sorted or not (in terms of sequence length)
# packed = rnn_utils.pack_sequence(seq_tensors, enforce_sorted=False)
# # data: [9,2]=>pack the first words of each sequence together, then pack the second words etc.
# # unsorted_indices: ([0,2,1])=>which sequences is the longest -> second longest -> etc. 
# # batch_sizes: ([3,3,2,1])=>3 sequences have >=1 word, 3 sequences have >=2 words, 2 squences have >=3 words, 1 sequence has >=4 words
# print("Packed sequence: ", packed) 
# print("Picking first sequence out: ", (packed.data[[0,3,6,8]] == seq_tensors[0]).all())
# output_packed, hidden_packed = rnn(packed)
# print("Output of packed sequence: ", output_packed) # PackedSequence; data:[9,2], unsorted_indices:([0,2,1]), batch_sizes:([3,3,2,1])
# print("Final hidden state of packed sequence: ", hidden_packed) # [1,3,2]
# # Notice: we cannot simply permute to get the final hidden of each sequence, but UNPACKING
# print("Unpack the shorted sequence hidden manually: ", output_packed.data[[2,5]]) # as 2nd and 5th words are of the shortest sequence - read rule of 'data' above
# output_unpacked, seq_sizes = rnn_utils.pad_packed_sequence(output_packed, batch_first=True)
# print("Unpacked output: ", output_unpacked) # [3,4,2]; Notice: the 2nd output sequence will have last 2 hiddens of zeros because those are padded words
# print("Sequences length: ", seq_sizes) # ([4,2,3])
# seq_idx = torch.arange(seq_sizes.size(0))
# print("Final hidden states of 3 sequences: ", output_unpacked[seq_idx, seq_sizes-1]) # [3,2]

# # To convert padded sequence to packed sequence
# len_seqs = [len(seq) for seq in all_seqs]
# packed = rnn_utils.pack_padded_sequence(
#     padded,
#     len_seqs,
#     enforce_sorted = False,
#     batch_first = True 
# )
# print("Packed sequence: ", packed)

# ## Variable-length dataset 
# var_points, var_directions = generate_sequences(variable_len=True) # list of 128 elements, each with shape [2,2] or [3,2] or [4,2]

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = [torch.as_tensor(s).float() for s in x]
        self.y = torch.as_tensor(y).float().view(-1,1)
    
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
    def __len__(self):
        return len(self.x)

# # train_var_data = CustomDataset(var_points, var_directions)
# # We cannot use DataLoader directly, because each sample has a different length
# # => Solution: packing with Collate Function i.e. telling how to batch data
# def pack_collate(batch): # batch: [(x_tensor,y_tensor), (x_tensor,y_tensor), ()]
#     X = [item[0] for item in batch]
#     y = [item[1] for item in batch]
#     X_pack = rnn_utils.pack_sequence(X, enforce_sorted=False)
#     return X_pack, torch.as_tensor(y).view(-1, 1)
# # Test the pack_collate function
# dummy_batch = [train_var_data[0], train_var_data[1]] # list of 2 tuples
# dummy_x, dummy_y = pack_collate(dummy_batch)
# print("Packed sequence X and tensor y:\n", dummy_x, '\n', dummy_y)
# train_var_loader = DataLoader(train_var_data, batch_size=2, shuffle=True, collate_fn=pack_collate)
# x_batch, y_batch = next(iter(train_var_loader))
# print("Batch of variable-length sequence:\n", x_batch, '\n', y_batch)

# class SquareModelPacked(nn.Module):
#     def __init__(self, n_features, hidden_dim, n_outputs):
#         super(SquareModelPacked, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.n_features = n_features
#         self.n_outputs = n_outputs
#         self.hidden = None 
#         self.cell = None 
#         self.basic_rnn = nn.LSTM(self.n_features, self.hidden_dim, bidirectional=True) # bidirectional LSTM
#         self.classifier = nn.Linear(2*self.hidden_dim, self.n_outputs)

#     # X: packed sequence
#     # final hidden state and final cell state: [2,N,H] (bidirectional LSTM)
#     def forward(self, X):
#         rnn_out, (self.hidden, self.cell) = self.basic_rnn(X)
#         # unpack padded output: [N,L,2*H]
#         batch_first_output, seq_sizes = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True)
#         seq_idx = torch.arange(seq_sizes.size(0)) # [0,1]
#         last_output = batch_first_output[seq_idx, seq_sizes-1] # [[0,1], [3,1]]=>[2,2*H], 1st row for 1st sequence, 2nd for 2nd sequence 
#         out = self.classifier(last_output) #  [N,1,n_outputs]
#         return out.view(-1, self.n_outputs)

# torch.manual_seed(21)
# model = SquareModelPacked(n_features=2, hidden_dim=2, n_outputs=1)
# loss = nn.BCEWithLogitsLoss() 
# optimizer = optim.Adam(model.parameters(), lr=.01)
# sbs_packed = StepByStep(model, loss, optimizer)
# sbs_packed.set_loaders(train_var_loader)
# sbs_packed.train(100)
# fig = sbs_packed.plot_losses()
# print("Validation result:\n", StepByStep.loader_apply(train_var_loader, sbs_packed.correct))

# ## 1D Convolutions
# temperatures = np.array([5, 11, 15, 6, 5, 3, 3, 0, 0, 3, 4, 2, 1])
# size = 5 
# weight = torch.ones(size) * 0.2
# print("Conv1d result: ", F.conv1d( 
#     torch.as_tensor(temperatures).float().view(1,1,-1),
#     weight = weight.view(1,1,-1)))
# seqs = torch.as_tensor(points).float() # [N,L,F]
# seqs_length_last = seqs.permute(0,2,1) # [N,F,L]

# torch.manual_seed(17)
# conv_seq = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2, bias=False)
# print("Conv1d shape: ", conv_seq.weight.shape) # [1,2,2] 
# # NOTE: Shape of sequence [L] convolve with filter [l_i], padding p, dilation d => floor of ((l_i+2*p)-df+d-1)/s +1
# print("Conv1d result: ", conv_seq(seqs_length_last[0:1])) # [1,1,3]

# # Dilation - if dilation=2 means takes the 0th and 2nd value to multiply filter, then 1st and 3rd...
# torch.manual_seed(17)
# conv_dilated = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2, dilation=2, bias=False)
# print("Dilated Conv1d shape: ", conv_dilated.weight.shape) # [1,2,2]
# print("Dilated Conv1d result: ", conv_dilated(seqs_length_last[0:1])) # [1,1,2]

# train_data = TensorDataset(
#     torch.as_tensor(points).float().permute(0, 2, 1), # [N,L,F]=>[N,F,L]
#     torch.as_tensor(directions).view(-1,1).float()
# )
# test_data = TensorDataset(
#     torch.as_tensor(test_points).float().permute(0,2,1),
#     torch.as_tensor(test_directions).view(-1,1).float()
# )
# train_loader = DataLoader(
#     train_data, batch_size=16, shuffle=True
# )
# test_loader = DataLoader(
#     test_data, batch_size=16
# )

# torch.manual_seed(21)
# model = nn.Sequential()
# model.add_module('conv1d', nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2))
# model.add_module('relu', nn.ReLU())
# model.add_module('flatten', nn.Flatten())
# model.add_module('output', nn.Linear(3,1))
# loss = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# sbs_conv1 = StepByStep(model, loss, optimizer)
# sbs_conv1.set_loaders(train_loader, test_loader)
# sbs_conv1.train(100)
# fig = sbs_conv1.plot_losses()
# plt.savefig('test.png')
# print("Model with Conv1d validation result: ", StepByStep.loader_apply(test_loader, sbs_conv1.correct))

## Putting all together
# Fixed length dataset
points, directions = generate_sequences(n=128, seed=13)
train_data = TensorDataset(
    torch.as_tensor(np.array(points)).float(),
    torch.as_tensor(directions).view(-1,1).float()
)
train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True 
)
# Variable length dataset
var_points, var_directions = generate_sequences(variable_len=True)
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = [torch.as_tensor(s).float() for s in x]
        self.y = torch.as_tensor(y).float().view(-1,1)
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    def __len__(self):
        return len(self.x)
train_var_data = CustomDataset(var_points, var_directions)

def pack_collate(batch):
    X = [item[0] for item in batch]
    y = [item[1] for item in batch]
    X_pack = rnn_utils.pack_sequence(X, enforce_sorted=False)
    return X_pack, torch.as_tensor(y).view(-1,1)

train_var_loader = DataLoader(
    train_var_data,
    batch_size=16,
    shuffle=True,
    collate_fn=pack_collate 
    )

class SquareModelOne(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs, rnn_layer=nn.LSTM, **kwargs):
        super(SquareModelOne, self).__init__()
        self.hidden_dim = hidden_dim 
        self.n_features = n_features 
        self.n_outputs = n_outputs 
        self.hidden = None 
        self.cell = None 
        self.basic_rnn = rnn_layer(self.n_features, self.hidden_dim, batch_first=True, **kwargs)
        output_dim = (self.basic_rnn.bidirectional + 1) * self.hidden_dim 
        self.classifier = nn.Linear(output_dim, self.n_outputs)

    def forward(self, X):
        is_packed = isinstance(X, nn.utils.rnn.PackedSequence)
        rnn_out, self.hidden = self.basic_rnn(X)
        if isinstance(self.basic_rnn, nn.LSTM):
            self.hidden, self.cell = self.hidden 
        if is_packed:
            # unpack the output (from PackedSequence to tensor of padded sequence)
            batch_first_output, seq_sizes = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True) # [N,L,H]
            seq_slice = torch.arange(seq_sizes.size(0)) # [0,1,2]
        else:
            batch_first_output = rnn_out # [N,L,H]
            seq_sizes = 0 # so the next step will take element -1 i.e. last output
            seq_slice = slice(None, None, None) # same as [:]
        last_output = batch_first_output[seq_slice, seq_sizes-1] # [N,1,H]
        out = self.classifier (last_output) # [N,1,n_outputs]
        return out.view(-1, self.n_outputs) # [N, n_outputs]

torch.manual_seed(21)
model = SquareModelOne(n_features=2, hidden_dim=2, n_outputs=1, rnn_layer=nn.LSTM, num_layers=1, bidirectional=True)
loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
sbs_one = StepByStep(model, loss, optimizer)
# sbs_one.set_loaders(train_loader)
sbs_one.set_loaders(train_var_loader)
sbs_one.train(100)
# print("Validation result of fixed-length dataset:\n", StepByStep.loader_apply(train_loader, sbs_one.correct))
print("Validation result of variable-length dataset:\n", StepByStep.loader_apply(train_var_loader, sbs_one.correct))