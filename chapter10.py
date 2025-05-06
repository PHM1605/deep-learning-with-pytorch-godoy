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
from seq2seq import PositionalEncoding, subsequent_mask, EncoderDecoderSelfAttn
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
    
## Layer Normalization - normalize rows i.e. mean and std over input-dimension-D
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

## Batch vs Layer
torch.manual_seed(23)
dummy_points = torch.randn(4,1,256)
dummy_pe = PositionalEncoding(1,256)
dummy_enc = dummy_pe(dummy_points) # [4,1,256]
fig = hist_encoding(dummy_enc)
plt.savefig('test.png')

layer_normalizer = nn.LayerNorm(256)
dummy_normed = layer_normalizer(dummy_enc)
print("Encoder normed: ", dummy_normed)
fig = hist_layer_normed(dummy_enc, dummy_normed)
plt.savefig('test.png')

## Our Seq2Seq problem
pe = PositionalEncoding(max_len=2, d_model=2)
source_seq = torch.tensor([
    [[1.0349, 0.9661],
    [0.8055, -0.9169]]
    ]) # [1,2,2]
source_seq_enc = pe(source_seq)
norm = nn.LayerNorm(2)
# notice: normalize 2 vectors => -1 or 1 only => not good => must increase Dimension with Projection/Embeddings
print("Norm of PE of Seq2Seq source:\n", norm(source_seq_enc))

## Projections (for numerical values) or Embeddings (for categorical)
torch.manual_seed(11)
proj_dim = 6 
linear_proj = nn.Linear(2, proj_dim)
pe = PositionalEncoding(2, proj_dim)
source_seq_proj = linear_proj(source_seq)
source_seq_proj_enc = pe(source_seq_proj)
norm = nn.LayerNorm(proj_dim)
print("Norm of PE of 6D source seq:\n", norm(source_seq_proj_enc))

class EncoderDecoderTransf(EncoderDecoderSelfAttn):
    def __init__(self, encoder, decoder, input_len, target_len, n_features):
        super(EncoderDecoderTransf, self).__init__(encoder, decoder, input_len, target_len)
        self.n_features = n_features 
        self.proj = nn.Linear(n_features, encoder.d_model)
        self.linear = nn.Linear(encoder.d_model, n_features)
    
    def encode(self, source_seq, source_mask=None):
        source_proj = self.proj(source_seq)
        encoder_states = self.encoder(source_proj, source_mask)
        self.decoder.init_keys(encoder_states)
    
    # during TRAINING
    def decode(self, shifted_target_seq, source_mask=None, target_mask=None):
        target_proj = self.proj(shifted_target_seq)
        outputs = self.decoder(target_proj, source_mask=source_mask, target_mask=target_mask)
        outputs = self.linear(outputs)
        return outputs 

## Data preparation
points, directions = generate_sequences(n=256, seed=13)
full_train = torch.as_tensor(np.array(points)).float()
target_train = full_train[:, 2:]
test_points, test_directions = generate_sequences(seed=19)
full_test = torch.as_tensor(np.array(test_points)).float()
source_test = full_test[:, :2]
target_test = full_test[:, 2:]
train_data = TensorDataset(full_train, target_train)
test_data = TensorDataset(source_test, target_test)
generator = torch.Generator()
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, generator=generator)
test_loader = DataLoader(test_data, batch_size=16)

fig = plot_data(points, directions, n_rows=1)
plt.savefig('test.png')

## Model configuration & training 
torch.manual_seed(42)
enclayer = EncoderLayer(n_heads=3, d_model=6, ff_units=10, dropout=0.1)
declayer = DecoderLayer(n_heads=3, d_model=6, ff_units=10, dropout=0.1)
enctransf = EncoderTransf(enclayer, n_layers=2)
dectransf = DecoderTransf(declayer, n_layers=2)
model_transf = EncoderDecoderTransf(
    enctransf, dectransf, input_len=2, target_len=2, n_features=2
)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model_transf.parameters(), lr=0.01)

for p in model_transf.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

sbs_seq_transf = StepByStep(model_transf, loss, optimizer)
sbs_seq_transf.set_loaders(train_loader, test_loader)
sbs_seq_transf.train(50)
fig = sbs_seq_transf.plot_losses()
plt.savefig('test.png')

## Train and val loss (we'll see val loss lower as training has dropout)
torch.manual_seed(11)
x, y = next(iter(train_loader))
device = sbs_seq_transf.device 
model_transf.train()
print("Training loss:\n", loss(model_transf(x.to(device)), y.to(device)))
model_transf.eval()
print("Val loss:\n", loss(model_transf(x.to(device)), y.to(device)))

fig = sequence_pred(sbs_seq_transf, full_test, test_directions)
plt.savefig('test.png')