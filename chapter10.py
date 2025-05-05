import copy 
import numpy as np 
import torch 
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset 
from torchvision.transforms.v2 import Compose, Normalize, Pad 

from data_generation.square_sequences import generate_sequences 
# from data_generation.image_classification import generate_dataset 
# from helpers import index_splitter, make_balanced_sampler 
from stepbystep.v4 import StepByStep 
from seq2seq import PositionalEncoding, subsequent_mask 
# from seq2seq import PositionalEncoding, subsequent_mask, EncoderDecoderSelfAttn 
from plots.chapter8 import *
from plots.chapter9 import *
from plots.chapter10 import *

## Narrow Attention 
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model 
        self.d_k = int(d_model/n_heads)
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.alphas = None 
    
    def make_chunks(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        # [N,L,D] => [N,L,n_heads, d_k]
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        # [N,n_heads,L,d_k]
        x = x.transpose(1,2)
        return x 
    
    def init_keys(self, key):
        self.proj_keys = self.make_chunks(self.linear_key(key)) # [N,n_heads,L,d_k]
        self.proj_value = self.make_chunks(self.linear_value(key)) # [N,n_heads,L,d_k]
    
    def score_function(self, query):
        # query: [N,L,D]
        proj_query = self.make_chunks(self.linear_query(query))
        # [N,n_heads,L,d_k]x[N,n_heads,d_k,L]=[N,n_heads,L,L]; use torch.matmul() instead of torch.bmm because there are 4D vectors (it will use last two dims only)
        dot_products = torch.matmul(
            proj_query, self.proj_keys.transpose(-2,-1)
        )
        scores = dot_products / np.sqrt(self.d_k)
        return scores 
    
    def attn(self, query, mask=None):
        scores = self.score_function(query) # [N,n_heads,L,L]
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        # [N,n_heads,L,L]; h0 will pay e.g. 0.8 attention to x0, 0.2 attention to x1
        alphas = F.softmax(scores, dim=-1) 
        alphas = self.dropout(alphas)
        self.alphas = alphas.detach()
        # [N,n_heads,L,L]x[N,n_heads,L,d_k]=[N,n_heads,L,d_k]
        context = torch.matmul(alphas, self.proj_value) 
        return context 

    def output_function(self, contexts):
        out = self.linear_out(contexts)
        return out 
    
    def forward(self, query, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # [N,1,L,L]; every head uses the same mask
        context = self.attn(query, mask=mask) # [N,n_heads,L,d_k]
        context = context.transpose(1,2).contiguous() # [N,L,n_heads,d_k]
        context = context.view(query.size(0),-1,self.d_model) # concatenating the context vectors; [N,L,n_heads*d_k]
        out = self.output_function(context) # [N,L,d_model]
        return out 

# Try on dummy points
dummy_points = torch.randn(16, 2, 4)
mha = MultiHeadedAttention(n_heads=2, d_model=4, dropout=0.0)
mha.init_keys(dummy_points)
out = mha(dummy_points) 
print("MultiHeadedAttention on dummy input: ", out.shape) #[16,2,4]

# in PyTorch: nn.TransformerEncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model # is the multiple of #heads
        self.ff_units = ff_units 
        self.self_attn_heads = MultiHeadedAttention(n_heads, d_model, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, d_model)
        )
        # Batch Normalization: normalize features; Layer Normalization: normalize data points 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, query, mask=None):
        # Sublayer #0 - Norm first
        norm_query = self.norm1(query)
        self.self_attn_heads.init_keys(norm_query)
        states = self.self_attn_heads(norm_query, mask)
        att = query + self.drop1(states)
        # Sublayer #1 - Norm first 
        norm_att = self.norm2(att)
        out = self.ffn(norm_att)
        out = att + self.drop2(out)
        return out 

# in PyTorch: nn.TransformerEncoder 
class EncoderTransf(nn.Module):
    # max_len: for PositionalEncoding
    def __init__(self, encoder_layer, n_layers=1, max_len=100):
        super().__init__()
        self.d_model = encoder_layer.d_model 
        self.pe = PositionalEncoding(max_len, self.d_model)
        self.norm = nn.LayerNorm(self.d_model) # final normalization at the end 
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer)
            for _ in range(n_layers)
        ])

    def forward(self, query, mask=None):
        x = self.pe(query)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# in PyTorch: nn.TransformerDecoderLayer 
class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads 
        self.d_model = d_model
        self.ff_units = ff_units 
        self.self_attn_heads = MultiHeadedAttention(n_heads, d_model, dropout)
        self.cross_attn_heads = MultiHeadedAttention(n_heads, d_model, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
    
    def init_keys(self, states):
        self.cross_attn_heads.init_keys(states)
    
    def forward(self, query, source_mask=None, target_mask=None):
        # Sublayer #0 - Norm-first
        norm_query = self.norm1(query)
        self.self_attn_heads.init_keys(norm_query)
        states = self.self_attn_heads(norm_query, target_mask)
        att1 = query + self.drop1(states)
        # Sublayer #1 - Norm-first
        norm_att1 = self.norm2(att1)
        encoder_states = self.cross_attn_heads(norm_att1, source_mask)
        att2 = att1 + self.drop2(encoder_states)
        # Sublayer #2 - Norm-first
        norm_att2 = self.norm3(att2)
        out = self.ffn(norm_att2)
        out = att2 + self.drop3(out)
        return out 

# in PyTorch: nn.TransformerDecoder 
class DecoderTransf(nn.Module):
    def __init__(self, decoder_layer, n_layers=1, max_len=100):
        super(DecoderTransf, self).__init__()
        self.d_model = decoder_layer.d_model 
        self.pe = PositionalEncoding(max_len, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer)
            for _ in range(n_layers)
        ])
    
    def init_keys(self, states):
        for layer in self.layers:
            layer.init_keys(states)
    
    def forward(self, query, source_mask=None, target_mask=None):
        x = self.pe(query)
        for layer in self.layers:
            x = layer(x, source_mask, target_mask)
        return self.norm(x)
    
## Layer Normalization - normalize rows i.e. each sample
d_model = 4
seq_len = 2
n_points = 3
torch.manual_seed(34)
data = torch.randn(n_points, seq_len, d_model)
pe = PositionalEncoding(seq_len, d_model)
inputs = pe(data) # [N,L,D]=[3,2,4]
print("Inputs shape: ", inputs.shape)
inputs_mean = inputs.mean(axis=2).unsqueeze(2)
print("Inputs mean of every sample:\n", inputs_mean) # [N,L,1]
inputs_var = inputs.var(axis=2, unbiased=False).unsqueeze(2)
print("Inputs var of every sample:\n", inputs_var) # [N,L,1]
print("Layer normalization manually:\n", (inputs-inputs_mean)/torch.sqrt(inputs_var+1e-5))
# Layer Normalization using PyTorch built-in
layer_norm = nn.LayerNorm(d_model)
normalized = layer_norm(inputs)
print("Layer normalization by library mean and std (1st sample only):\n", normalized[0][0].mean(), normalized[0][0].std(unbiased=False))
# Notice: LayerNorm learnable weight and bias don't interfere with input; but do the normalization calculation as a pre-effect
print("LayerNorm learnable weight and bias: ", layer_norm.state_dict())