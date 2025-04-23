import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset 
from data_generation.square_sequences import generate_sequences
from stepbystep.v4 import StepByStep 
# from plots.chapter8 import *
from plots.chapter9 import *

# fig = counter_vs_clock(binary=False)
# plt.savefig('test.png')

# fig = plot_sequences(binary=False, target_len=2)
# plt.savefig('test.png')

# points, directions = generate_sequences(n=256, seed=13)
# fig = plot_data(points, directions, n_rows=1)
# plt.savefig('test.png')

class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim 
        self.n_features = n_features
        self.hidden = None 
        self.basic_rnn = nn.GRU(
            self.n_features, 
            self.hidden_dim, 
            batch_first=True)
    def forward(self, X):
        rnn_out, self.hidden = self.basic_rnn(X)
        return rnn_out # [N,L,H]

# # Encoding dummy sequence
# full_seq = torch.tensor([[-1,-1],[-1,1],[1,1],[1,-1]]).float().view(1,4,2)
# source_seq = full_seq[:,:2] # [1,2,2]
# target_seq = full_seq[:,2:] # [1,2,2]

# torch.manual_seed(21)
# encoder = Encoder(n_features=2, hidden_dim=2)
# hidden_seq = encoder(source_seq) # [N,L,F] with L=2
# hidden_final = hidden_seq[:, -1:] # [N,1,F]

class Decoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim 
        self.n_features = n_features 
        self.hidden = None 
        self.basic_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True)
        self.regression = nn.Linear(self.hidden_dim, self.n_features)

    def init_hidden(self, hidden_seq):
        # We only need final hidden state
        hidden_final = hidden_seq[:,-1:] # [N,1,H]
        self.hidden = hidden_final.permute(1,0,2) # [1,N,H] - batch second
    # X is [N,1,F]
    # this 'forward' method will be called several times for each point in target sequence
    def forward(self, X):
        batch_first_output, self.hidden = self.basic_rnn(X, self.hidden)
        last_output = batch_first_output[:,-1:] # [N,1,F]
        out = self.regression(last_output)
        return out.view(-1,1,self.n_features) # [N,1,F]

# ## Choosing at random between Teacher Forcing (feeding real target sequence values) OR Feeding Previous Prediction
# torch.manual_seed(21)
# decoder = Decoder(n_features=2, hidden_dim=2)
# decoder.init_hidden(hidden_seq)
# inputs = source_seq[:,-1:] # [1,1,2]
# teacher_forcing_prob = 0.5
# target_len = 2
# for i in range(target_len):
#     print(f"Hidden: {decoder.hidden}")
#     out = decoder(inputs) # [1,1,2]
#     print(f"Output: {out}\n")
#     if torch.rand(1) <= teacher_forcing_prob:
#         inputs = target_seq[:, i:i+1] # Teacher forcing (feeding real target sequence values)
#     else:
#         inputs = out # Feeding previous prediction 

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, input_len, target_len, teacher_forcing_prob=0.5):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder
        self.input_len = input_len 
        self.target_len = target_len
        self.teacher_forcing_prob = teacher_forcing_prob 
        self.outputs = None 

    def init_outputs(self, batch_size):
        device = next(self.parameters()).device 
        # outputs: [N,L_target,F]
        self.outputs = torch.zeros(batch_size, self.target_len, self.encoder.n_features).to(device)
    
    def store_output(self, i, out):
        self.outputs[:,i:i+1,:] = out 

    # X: [N,L,F] - L: full length of sequence; will be splitted
    def forward(self, X):
        source_seq = X[:, :self.input_len, :] # [N,L_source,F]
        target_seq = X[:, self.input_len:, :] # [N,L_target,F]
        self.init_outputs(X.shape[0])
        hidden_seq = self.encoder(source_seq) # [N,L_source,H]
        self.decoder.init_hidden(hidden_seq)
        # The last input of the encoder is also the 1st input of the decoder
        dec_inputs = source_seq[:,-1:,:] # [N,1,F]
        for i in range(self.target_len):
            out = self.decoder(dec_inputs)
            self.store_output(i, out)
            prob = self.teacher_forcing_prob 
            if not self.training:
                prob = 0
            if torch.rand(1) <= prob:
                dec_inputs = target_seq[:, i:i+1, :] # take the real measured value (more accurate)
            else:
                dec_inputs = out # take the predicted value (less accurate, from our current model)
        return self.outputs 

# encdec = EncoderDecoder(encoder, decoder, input_len=2, target_len=2, teacher_forcing_prob=0.5)
# # in training mode, the model expects full sequence for teacher-forcing 
# encdec.train() # switch the 'model.train' property to True
# print("Naive prediction during training: ", encdec(full_seq))
# # in evaluation mode, the model expects only source sequence
# encdec.eval() # switch the 'model.train' property to False
# print("Naive prediction during evaluation: ", encdec(source_seq))

# ## Data preparation
# points, directions = generate_sequences(n=256, seed=13)
# full_train = torch.as_tensor(np.array(points)).float() # [256,4,2]
# target_train = full_train[:, 2:] # [256,2,2]

# test_points, test_directions = generate_sequences(seed=19)
# full_test = torch.as_tensor(np.array(test_points)).float()
# source_test = full_test[:, :2] #  [128,2,2]
# target_test = full_test[:, 2:] # [128,2,2]

# train_data = TensorDataset(full_train, target_train)
# test_data = TensorDataset(source_test, target_test)
# generator = torch.Generator()
# train_loader = DataLoader(train_data, batch_size=16, shuffle=True, generator=generator)
# test_loader = DataLoader(test_data, batch_size=16)

# ## Model configuration and training 
# torch.manual_seed(23)
# encoder = Encoder(n_features=2, hidden_dim=2)
# decoder = Decoder(n_features=2, hidden_dim=2)
# model = EncoderDecoder(encoder, decoder, input_len=2, target_len=2, teacher_forcing_prob=0.5)
# loss = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# sbs_seq = StepByStep(model, loss, optimizer)
# sbs_seq.set_loaders(train_loader, test_loader)
# sbs_seq.train(100)
# fig = sbs_seq.plot_losses()
# plt.savefig('test.png')

# ## Visualize predictions
# fig = sequence_pred(sbs_seq, full_test, test_directions)
# plt.savefig('test.png')

## Attention
# Computing the context vector 
full_seq = torch.tensor([[-1,-1],[-1,1],[1,1],[1,-1]]).float().view(1,4,2)
source_seq = full_seq[:,:2] # [1,2,2]
target_seq = full_seq[:,2:]

torch.manual_seed(21)
encoder = Encoder(n_features=2, hidden_dim=2)
hidden_seq = encoder(source_seq) # [N,L,H]=[1,2,2]
values = hidden_seq 
keys = hidden_seq 

torch.manual_seed(31)
decoder = Decoder(n_features=2, hidden_dim=2)
decoder.init_hidden(hidden_seq)
inputs = source_seq[:,-1:] # [N,1,F]=[1,1,2]
out = decoder(inputs)
query = decoder.hidden.permute(1,0,2) # hidden: [1,N,H]=>[N,1,H]

# dummy calculation
def calc_alphas(ks, q):
    N, L, H = ks.size()
    alphas = torch.ones(N,1,L).float() * 1/L
    return alphas

alphas = calc_alphas(keys, query) # [1,1,2]
context_vector = torch.bmm(alphas, values) # matrix-multiplication, [N,1,L]x[N,L,H]=[N,1,H]
concatenated = torch.cat([context_vector, query], axis=-1) # [N,1,2H]

# Scoring method: to check if two hidden vectors are similar or not => using COSINE
products = torch.bmm(query, keys.permute(0,2,1)) # [N,1,H]x[N,H,L]=[N,1,L]
alphas = F.softmax(products, dim=-1) # [N,1,L] (sum each row is 1)

def calc_alphas(ks, q):
    products = torch.bmm(q, ks.permute(0,2,1)) # [N,1,H]x[N,H,L]=[N,1,L]
    alphas = F.softmax(products, dim=-1)
    return alphas 

dims = query.size(-1) # H
scaled_products = products / np.sqrt(dims)

# Visualizing the context Vector
q = torch.tensor([0.55, 0.95]).view(1,1,2) # [N,1,H]
k = torch.tensor([[0.65, 0.2], [0.85, -0.4], [-0.95, -0.75]]).view(1,3,2) # [N,L,H]
fig = query_and_keys(q.squeeze(), k.view(3,2)) # [H], [L,H]