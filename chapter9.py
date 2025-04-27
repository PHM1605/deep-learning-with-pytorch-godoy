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

## Data preparation
points, directions = generate_sequences(n=256, seed=13)
full_train = torch.as_tensor(np.array(points)).float() # [256,4,2]
target_train = full_train[:, 2:] # [256,2,2]

test_points, test_directions = generate_sequences(seed=19)
full_test = torch.as_tensor(np.array(test_points)).float()
source_test = full_test[:, :2] #  [128,2,2]
target_test = full_test[:, 2:] # [128,2,2]

train_data = TensorDataset(full_train, target_train)
test_data = TensorDataset(source_test, target_test)
generator = torch.Generator()
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, generator=generator)
test_loader = DataLoader(test_data, batch_size=16)

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

# ## Attention
# fig = figure9()
# plt.savefig('test.png')
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
dims = query.size(-1)
products = torch.bmm(query, keys.permute(0,2,1)) # [N,1,H]x[N,H,L]=[N,1,L]
scaled_products = products / np.sqrt(dims)
alphas = F.softmax(scaled_products, dim=-1) # [N,1,L] (sum each row is 1)
print("Alphas: ", alphas)

def calc_alphas(ks, q):
    dims = q.size(-1) # H
    products = torch.bmm(q, ks.permute(0,2,1)) # [N,1,H]x[N,H,L]=[N,1,L]
    scaled_products = products / np.sqrt(dims)
    alphas = F.softmax(scaled_products, dim=-1) # [N,1,L]
    return alphas 

# Visualizing the context Vector with dummy q and k
q = torch.tensor([0.55, 0.95]).view(1,1,2) # [N,1,H]
k = torch.tensor([[0.65, 0.2], [0.85, -0.4], [-0.95, -0.75]]).view(1,3,2) # [N,L,H]
fig = query_and_keys(q.squeeze(), k.view(3,2)) # [H], [L,H]
plt.savefig('test.png')
# [N,1,H]x[N,H,L]=[N,1,L]
prod = torch.bmm(q, k.permute(0,2,1))
print("How query 0th fits the keys of source sequence: ", prod)
scores = F.softmax(prod, dim=-1)
print("Softmax score of query 0th with keys of source sequence: ", scores)
v = k 
context = torch.bmm(scores, v) # [N,1,L]x[N,L,H]=[N,1,H]
print("Context: ", context)
fig = query_and_keys(q.squeeze(), k.view(3,2), context)
plt.savefig('test.png')

# Continue
alphas = calc_alphas(keys, query)
context_vector = torch.bmm(alphas, values) # [N,1,L]x[N,L,H]=[N,1,H]
print("Context vector: ", context_vector)

class Attention(nn.Module):
    def __init__(self, hidden_dim, input_dim = None, proj_values=False):
        super().__init__()
        self.d_k = hidden_dim # dimension of key K, to scale the dot product between key and query
        self.input_dim = hidden_dim if input_dim is None else input_dim 
        self.proj_values = proj_values # to set we will project the value V or not

        self.linear_query = nn.Linear(self.input_dim, hidden_dim)
        self.linear_key = nn.Linear(self.input_dim, hidden_dim)
        self.linear_value = nn.Linear(self.input_dim, hidden_dim)
        self.alphas = None 
    
    # keys: [N,L,H]
    def init_keys(self, keys):
        self.keys = keys 
        self.proj_keys = self.linear_key(self.keys)
        self.values = self.linear_value(self.keys) if self.proj_values else self.keys
    
    def score_function(self, query):
        proj_query = self.linear_query(query)
        # [N,1,H]x[N,H,L]->[N,1,L]
        dot_products = torch.bmm(proj_query, self.proj_keys.permute(0,2,1))
        scores = dot_products / np.sqrt(self.d_k)
        return scores # [N,1,L]

    def forward(self, query, mask=None):
        # query: [N,1,L]; scores: [N,1,H]
        scores = self.score_function(query)
        if mask is not None:            
            # set scores of padding sample [0,0] to minus infinity
            scores = scores.masked_fill(mask==0, -1e9) # [N,1,L]=[[[False,True]]]
        alphas = F.softmax(scores, dim=-1) # [N,1,L]
        self.alphas = alphas.detach()
        # [N,1,L]x[N,L,H]->[N,1,H]
        context = torch.bmm(alphas, self.values)
        return context 

## Source mask - mask some (padded) values V to ignore them
source_seq = torch.tensor([[[-1,1], [0,0]]]) # [N,L,H]=[1,2,2]
# ..assuming there is an encoder here
keys = torch.tensor([[[-0.38, 0.44], [0.85, -0.05]]]) # [N,L,H]
query = torch.tensor([[[-1., 1.]]]) # [N,1,H]=[1,1,2]
source_mask = (source_seq != 0).all(axis=2).unsqueeze(1) # [N,1,L]
print("Source mask: ", source_mask) # [[[True, False]]]
torch.manual_seed(11)
attnh = Attention(2) # hidden_dim = input_dim = 2
attnh.init_keys(keys) # init key and value
context = attnh(query, mask=source_mask)
print("Attention alphas: ", attnh.alphas)

class DecoderAttn(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim 
        self.n_features = n_features 
        self.hidden = None # hidden final from Encoder
        self.basic_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True)
        self.attn = Attention(self.hidden_dim)
        # regression just before the Decoder's output
        self.regression = nn.Linear(2*self.hidden_dim, self.n_features)
    
    # hidden_seq: [N,L,H]
    def init_hidden(self, hidden_seq):
        self.attn.init_keys(hidden_seq) # init & project key and value
        hidden_final = hidden_seq[:, -1:] # [N,1,H]
        self.hidden = hidden_final.permute(1,0,2) # [L,N,H] with L=1

    # X: [N,1,F]
    def forward(self, X, mask=None):
        # [N,1,H] for both
        batch_first_output, self.hidden = self.basic_rnn(X, self.hidden)
        query = batch_first_output[:, -1:]
        context = self.attn(query, mask=mask) # [N,1,H]
        concatenated = torch.cat([context, query], axis=-1) # [N,1,2*H]
        out = self.regression(concatenated) # [N,1,n_features]
        return out.view(-1,1,self.n_features)

full_seq = torch.tensor([[-1,-1],[-1,1],[1,1],[1,-1]]).float().view(1,4,2)
source_seq = full_seq[:,:2] # [N,L,F]=[1,2,2]
target_seq = full_seq[:,2:] # [1,2,2]
torch.manual_seed(21)
encoder = Encoder(n_features=2, hidden_dim=2)
decoder_attn = DecoderAttn(n_features=2, hidden_dim=2)
hidden_seq = encoder(source_seq)
decoder_attn.init_hidden(hidden_seq)
# Target sequence generation
inputs = source_seq[:,-1:] # [N,L,H]=[1,1,2]
target_len = 2 
for i in range(target_len):
    out = decoder_attn(inputs)
    print(f'Output: {out}')
    inputs = out 

## Encoder + Decoder + Attention
encdec = EncoderDecoder(encoder, decoder_attn, 
    input_len=2, target_len=2, teacher_forcing_prob=0.0)
print("Encoder Decoder Attention result: ", encdec(full_seq))

## Still, create a class inheriting from above, but with storing attention scores
class EncoderDecoderAttn(EncoderDecoder):
    def __init__(self, encoder, decoder, input_len, target_len, teacher_forcing_prob=0.5):
        super().__init__(encoder, decoder, input_len, target_len, teacher_forcing_prob)
        self.alphas = None 
    
    # overwrite the "init_outputs" and "store_output"
    def init_outputs(self, batch_size):
        device = next(self.parameters()).device
        # [N,L(target),F] 
        self.outputs = torch.zeros(
            batch_size, self.target_len, self.encoder.n_features
        ).to(device)
        # [N,L(target),L(source)]
        self.alphas = torch.zeros(
            batch_size, self.target_len, self.input_len
        ).to(device)
    
    def store_output(self, i, out):
        self.outputs[:, i:i+1, :] = out
        self.alphas[:, i:i+1, :] = self.decoder.attn.alphas 

torch.manual_seed(17)
encoder = Encoder(n_features=2, hidden_dim=2)
decoder_attn = DecoderAttn(n_features=2, hidden_dim=2)
model = EncoderDecoderAttn(encoder, decoder_attn, input_len=2, target_len=2, teacher_forcing_prob=0.5)
loss = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01)
sbs_seq_attn = StepByStep(model, loss, optimizer)
sbs_seq_attn.set_loaders(train_loader, test_loader)
sbs_seq_attn.train(100)
fig = sbs_seq_attn.plot_losses()
plt.savefig('test.png')

# Visualizing Predictions
fig = sequence_pred(sbs_seq_attn, full_test, test_directions)
plt.savefig('test.png')

# Visualizing Attention
# First sample
inputs = full_train[:1, :2] # [1,2,2]
out = sbs_seq_attn.predict(inputs)
print("Attention score of first sample:\n", sbs_seq_attn.model.alphas)
# First 10 samples 
inputs = full_train[:10, :2] # [10,2,2]
source_labels = ['Point #1', 'Point #2']
target_labels = ['Point #3', 'Point #4']
point_labels = [
    f'{"Counter-" if not directions[i] else ""}Clockwise\nPoint #1: {inp[0,0]:.2f},{inp[0,1]:.2f}'
    for i, inp in enumerate(inputs)
]
fig = plot_attention(model, inputs, point_labels, source_labels, target_labels)
plt.savefig('test.png')
