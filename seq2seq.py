import numpy as np
import torch 
import torch.optim as optim 
import torch.nn as nn 
import torch.nn.functional as F 

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = (1-torch.triu(torch.ones(attn_shape), diagonal=1)).bool()
    return subsequent_mask 

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1) #[L,1]
        angular_speed = torch.exp(torch.arange(0,d_model,2).float()*(-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*angular_speed)
        pe[:, 1::2] = torch.cos(position*angular_speed)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1,max_len,D]
    
    def forward(self, x):
        # x: [N,L,D]; pe: [1,max_len,D]
        scaled_x = x*np.sqrt(self.d_model)
        encoded = scaled_x + self.pe[:, :x.size(1),:] # [N,L,D]
        return encoded 

