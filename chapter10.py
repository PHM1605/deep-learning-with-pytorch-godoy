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
        