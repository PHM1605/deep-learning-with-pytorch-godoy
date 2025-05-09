import copy 
import numpy as np 
import torch 
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset 
from torchvision.transforms.v2 import Compose, Normalize, Pad 

from data_generation.square_sequences import generate_sequences 
from data_generation.image_classification import generate_dataset 
from helpers import index_splitter, make_balanced_sampler 
from stepbystep.v4 import StepByStep 
from seq2seq import PositionalEncoding, subsequent_mask, EncoderDecoderSelfAttn, EncoderLayer, DecoderLayer, EncoderTransf, DecoderTransf
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

# # Try on dummy points
# dummy_points = torch.randn(16, 2, 4)
# mha = MultiHeadedAttention(n_heads=2, d_model=4, dropout=0.0)
# mha.init_keys(dummy_points)
# out = mha(dummy_points) 
# print("MultiHeadedAttention on dummy input: ", out.shape) #[16,2,4]

# # in PyTorch: nn.TransformerEncoderLayer
# class EncoderLayer(nn.Module):
#     def __init__(self, n_heads, d_model, ff_units, dropout=0.1):
#         super().__init__()
#         self.n_heads = n_heads
#         self.d_model = d_model # is the multiple of #heads
#         self.ff_units = ff_units 
#         self.self_attn_heads = MultiHeadedAttention(n_heads, d_model, dropout)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, ff_units),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(ff_units, d_model)
#         )
#         # Batch Normalization: normalize features; Layer Normalization: normalize data points 
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.drop1 = nn.Dropout(dropout)
#         self.drop2 = nn.Dropout(dropout)
    
#     def forward(self, query, mask=None):
#         # Sublayer #0 - Norm first
#         norm_query = self.norm1(query)
#         self.self_attn_heads.init_keys(norm_query)
#         states = self.self_attn_heads(norm_query, mask)
#         att = query + self.drop1(states)
#         # Sublayer #1 - Norm first 
#         norm_att = self.norm2(att)
#         out = self.ffn(norm_att)
#         out = att + self.drop2(out)
#         return out 

# # in PyTorch: nn.TransformerEncoder 
# # norm-first for each sublayer
# class EncoderTransf(nn.Module):
#     # max_len: for PositionalEncoding
#     def __init__(self, encoder_layer, n_layers=1, max_len=100):
#         super().__init__()
#         self.d_model = encoder_layer.d_model 
#         self.pe = PositionalEncoding(max_len, self.d_model)
#         self.norm = nn.LayerNorm(self.d_model) # final normalization at the end 
#         self.layers = nn.ModuleList([
#             copy.deepcopy(encoder_layer)
#             for _ in range(n_layers)
#         ])

#     def forward(self, query, mask=None):
#         x = self.pe(query)
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)

# # in PyTorch: nn.TransformerDecoderLayer 
# # norm-first for each sublayer
# class DecoderLayer(nn.Module):
#     def __init__(self, n_heads, d_model, ff_units, dropout=0.1):
#         super().__init__()
#         self.n_heads = n_heads 
#         self.d_model = d_model
#         self.ff_units = ff_units 
#         self.self_attn_heads = MultiHeadedAttention(n_heads, d_model, dropout)
#         self.cross_attn_heads = MultiHeadedAttention(n_heads, d_model, dropout)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, ff_units),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(ff_units, d_model)
#         )
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.drop1 = nn.Dropout(dropout)
#         self.drop2 = nn.Dropout(dropout)
#         self.drop3 = nn.Dropout(dropout)
    
#     def init_keys(self, states):
#         self.cross_attn_heads.init_keys(states)
    
#     def forward(self, query, source_mask=None, target_mask=None):
#         # Sublayer #0 - Norm-first
#         norm_query = self.norm1(query)
#         self.self_attn_heads.init_keys(norm_query)
#         states = self.self_attn_heads(norm_query, target_mask)
#         att1 = query + self.drop1(states)
#         # Sublayer #1 - Norm-first
#         norm_att1 = self.norm2(att1)
#         encoder_states = self.cross_attn_heads(norm_att1, source_mask)
#         att2 = att1 + self.drop2(encoder_states)
#         # Sublayer #2 - Norm-first
#         norm_att2 = self.norm3(att2)
#         out = self.ffn(norm_att2)
#         out = att2 + self.drop3(out)
#         return out 

# # in PyTorch: nn.TransformerDecoder 
# class DecoderTransf(nn.Module):
#     def __init__(self, decoder_layer, n_layers=1, max_len=100):
#         super(DecoderTransf, self).__init__()
#         self.d_model = decoder_layer.d_model 
#         self.pe = PositionalEncoding(max_len, self.d_model)
#         self.norm = nn.LayerNorm(self.d_model) # at output
#         self.layers = nn.ModuleList([
#             copy.deepcopy(decoder_layer)
#             for _ in range(n_layers)
#         ])
    
#     def init_keys(self, states):
#         for layer in self.layers:
#             layer.init_keys(states)
    
#     def forward(self, query, source_mask=None, target_mask=None):
#         x = self.pe(query)
#         for layer in self.layers:
#             x = layer(x, source_mask, target_mask)
#         return self.norm(x)
    
# ## Layer Normalization - normalize rows i.e. mean and std over input-dimension-D
# d_model = 4
# seq_len = 2
# n_points = 3
# torch.manual_seed(34)
# data = torch.randn(n_points, seq_len, d_model)
# pe = PositionalEncoding(seq_len, d_model)
# inputs = pe(data) # [N,L,D]=[3,2,4]
# print("Inputs shape: ", inputs.shape)
# inputs_mean = inputs.mean(axis=2).unsqueeze(2)
# print("Inputs mean of every sample:\n", inputs_mean) # [N,L,1]
# inputs_var = inputs.var(axis=2, unbiased=False).unsqueeze(2)
# print("Inputs var of every sample:\n", inputs_var) # [N,L,1]
# print("Layer normalization manually:\n", (inputs-inputs_mean)/torch.sqrt(inputs_var+1e-5))
# # Layer Normalization using PyTorch built-in
# layer_norm = nn.LayerNorm(d_model)
# normalized = layer_norm(inputs)
# print("Layer normalization by library mean and std (1st sample only):\n", normalized[0][0].mean(), normalized[0][0].std(unbiased=False))
# # Notice: LayerNorm learnable weight and bias don't interfere with input; but do the normalization calculation as a pre-effect
# print("LayerNorm learnable weight and bias: ", layer_norm.state_dict())

# ## Batch vs Layer
# torch.manual_seed(23)
# dummy_points = torch.randn(4,1,256)
# dummy_pe = PositionalEncoding(1,256)
# dummy_enc = dummy_pe(dummy_points) # [4,1,256]
# fig = hist_encoding(dummy_enc)
# plt.savefig('test.png')

# layer_normalizer = nn.LayerNorm(256)
# dummy_normed = layer_normalizer(dummy_enc)
# print("Encoder normed: ", dummy_normed)
# fig = hist_layer_normed(dummy_enc, dummy_normed)
# plt.savefig('test.png')

# ## Our Seq2Seq problem
# pe = PositionalEncoding(max_len=2, d_model=2)
# source_seq = torch.tensor([
#     [[1.0349, 0.9661],
#     [0.8055, -0.9169]]
#     ]) # [1,2,2]
# source_seq_enc = pe(source_seq)
# norm = nn.LayerNorm(2)
# # notice: normalize 2 vectors => -1 or 1 only => not good => must increase Dimension with Projection/Embeddings
# print("Norm of PE of Seq2Seq source:\n", norm(source_seq_enc))

# ## Projections (for numerical values) or Embeddings (for categorical)
# torch.manual_seed(11)
# proj_dim = 6 
# linear_proj = nn.Linear(2, proj_dim)
# pe = PositionalEncoding(2, proj_dim)
# source_seq_proj = linear_proj(source_seq)
# source_seq_proj_enc = pe(source_seq_proj)
# norm = nn.LayerNorm(proj_dim)
# print("Norm of PE of 6D source seq:\n", norm(source_seq_proj_enc))

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

# ## Data preparation
# points, directions = generate_sequences(n=256, seed=13)
# full_train = torch.as_tensor(np.array(points)).float()
# target_train = full_train[:, 2:]
# test_points, test_directions = generate_sequences(seed=19)
# full_test = torch.as_tensor(np.array(test_points)).float()
# source_test = full_test[:, :2]
# target_test = full_test[:, 2:]
# train_data = TensorDataset(full_train, target_train)
# test_data = TensorDataset(source_test, target_test)
# generator = torch.Generator()
# train_loader = DataLoader(train_data, batch_size=16, shuffle=True, generator=generator)
# test_loader = DataLoader(test_data, batch_size=16)

# fig = plot_data(points, directions, n_rows=1)
# plt.savefig('test.png')

# ## Model configuration & training 
# torch.manual_seed(42)
# enclayer = EncoderLayer(n_heads=3, d_model=6, ff_units=10, dropout=0.1)
# declayer = DecoderLayer(n_heads=3, d_model=6, ff_units=10, dropout=0.1)
# enctransf = EncoderTransf(enclayer, n_layers=2)
# dectransf = DecoderTransf(declayer, n_layers=2)
# model_transf = EncoderDecoderTransf(
#     enctransf, dectransf, input_len=2, target_len=2, n_features=2
# )
# loss = nn.MSELoss()
# optimizer = torch.optim.Adam(model_transf.parameters(), lr=0.01)

# for p in model_transf.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)

# sbs_seq_transf = StepByStep(model_transf, loss, optimizer)
# sbs_seq_transf.set_loaders(train_loader, test_loader)
# sbs_seq_transf.train(50)
# fig = sbs_seq_transf.plot_losses()
# plt.savefig('test.png')

# ## Train and val loss (we'll see val loss lower as training has dropout)
# torch.manual_seed(11)
# x, y = next(iter(train_loader))
# device = sbs_seq_transf.device 
# model_transf.train()
# print("Training loss:\n", loss(model_transf(x.to(device)), y.to(device)))
# model_transf.eval()
# print("Val loss:\n", loss(model_transf(x.to(device)), y.to(device)))

# fig = sequence_pred(sbs_seq_transf, full_test, test_directions)
# plt.savefig('test.png')

## PyTorch Transformer model
class TransformerModel(nn.Module):
    def __init__(self, transformer, input_len, target_len, n_features):
        super().__init__()
        self.transf = transformer 
        self.input_len = input_len 
        self.target_len = target_len 
        self.trg_masks = self.transf.generate_square_subsequent_mask(self.target_len)
        self.n_features = n_features 
        self.proj = nn.Linear(n_features, self.transf.d_model) # at input 
        self.linear = nn.Linear(self.transf.d_model, n_features) # at target
        max_len = max(self.input_len, self.target_len)
        self.pe = PositionalEncoding(max_len, self.transf.d_model)
        self.norm = nn.LayerNorm(self.transf.d_model)
    
    def preprocess(self, seq):
        seq_proj = self.proj(seq)
        seq_enc = self.pe(seq_proj)
        return self.norm(seq_enc)
    
    # during TRAIN
    def encode_decode(self, source, target, source_mask=None, target_mask=None):
        # Projections
        src = self.preprocess(source)
        tgt = self.preprocess(target)
        out = self.transf(src, tgt, src_key_padding_mask=source_mask, tgt_mask=target_mask)
        out = self.linear(out) # [N,L,F]
        return out 
    
    # during VAL/TEST
    def predict(self, source_seq, source_mask=None):
        inputs = source_seq[:, -1:]
        for i in range(self.target_len):
            out = self.encode_decode(
                source_seq, inputs, source_mask=source_mask, target_mask=self.trg_masks[:i+1,:i+1]
                )
            out = torch.cat([inputs, out[:,-1:, :]], dim=-2)
            inputs = out.detach()
        outputs = out[:,1:,:]
        return outputs 
    
    def forward(self, X, source_mask=None):
        self.trg_masks= self.trg_masks.type_as(X)
        source_seq = X[:,:self.input_len,:]
        if self.training:
            shifted_target_seq = X[:, self.input_len-1:-1, :]
            outputs = self.encode_decode(
                source_seq, shifted_target_seq, source_mask=source_mask, target_mask=self.trg_masks
            )
        else:
            outputs = self.predict(source_seq, source_mask)
        return outputs 

# torch.manual_seed(42)
# transformer = nn.Transformer(
#     d_model=6,
#     nhead=3,
#     num_encoder_layers=1,
#     num_decoder_layers=1,
#     dim_feedforward=20,
#     dropout = 0.1,
#     batch_first = True
# )
# model_transformer = TransformerModel(transformer, input_len=2, target_len=2, n_features=2)
# loss = nn.MSELoss()
# optimizer = torch.optim.Adam(model_transformer.parameters(), lr=0.01)
# # Weight init
# for p in model_transformer.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)

# sbs_seq_transformer = StepByStep(
#     model_transformer, loss, optimizer
# )
# sbs_seq_transformer.set_loaders(train_loader, test_loader)
# sbs_seq_transformer.train(50)
# fig = sbs_seq_transformer.plot_losses()
# plt.savefig('test.png')

# ## Vision Transformer 
# # Data generation & preparation
# images, labels = generate_dataset(img_size=12, n_images=1000, binary=False, seed=17) # [1000,1,12,12]
# img = torch.as_tensor(images[2]).unsqueeze(0).float()/255 # [num_images, color_channel, height, width] = [1,1,12,12]
# fig = plot_images(img, title=False)
# plt.savefig('test.png')

class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x 
        self.y = y 
        self.transform = transform 
    
    def __getitem__(self, index):
        x = self.x[index]
        if self.transform:
            x = self.transform(x)
        return x, self.y[index]
    
    def __len__(self):
        return len(self.x)

# x_tensor = torch.as_tensor(images/255).float()
# y_tensor = torch.as_tensor(labels).long()
# train_idx, val_idx = index_splitter(len(x_tensor), [80, 20])
# x_train_tensor = x_tensor[train_idx]
# y_train_tensor = y_tensor[train_idx]
# x_val_tensor = x_tensor[val_idx]
# y_val_tensor = y_tensor[val_idx]
# train_composer = Compose([Normalize(mean=(0.5,), std=(0.5,))])
# val_composer = Compose([Normalize(mean=(0.5,), std=(0.5,))])
# train_dataset = TransformedTensorDataset(
#     x_train_tensor, y_train_tensor, transform=train_composer
# )
# val_dataset = TransformedTensorDataset(
#     x_val_tensor, y_val_tensor, transform=val_composer 
# )
# sampler = make_balanced_sampler(y_train_tensor)
# train_loader = DataLoader(
#     dataset = train_dataset, batch_size=16, sampler=sampler 
# )
# val_loader = DataLoader(
#     dataset=val_dataset, batch_size=16
# )

def extract_image_patches(x, kernel_size, stride=1):
    patches = x.unfold(2, kernel_size, stride) # [1,1,12,12]=>[1,1,3,4,12]
    patches = patches.unfold(3, kernel_size, stride) # [1,1,3,4,12]=>[1,1,3,3,4,4] with [3,3]=[num_vertical_patches,num_horizontal_patches]
    patches = patches.permute(0,2,3,1,4,5).contiguous() # [num_images,num_vertical,num_horizontal,num_channels,patch_height,patch_width]
    return patches.view(x.shape[0], patches.shape[1], patches.shape[2], -1) # [num_images,num_vertical,num_horizontal,num_elements_in_kernel]=[1,3,3,16]

# kernel_size = 4
# # img = [num_images,color_channel,height,width] = [1,1,12,12]
# patches = extract_image_patches(
#     img, kernel_size, stride=kernel_size
# )
# print("Patches shape:\n", patches.shape) # [1,3,3,16]
# fig = plot_patches(patches, kernel_size=kernel_size)
# plt.savefig('test.png')

# seq_patches = patches.view(-1, patches.size(-1)) # [9,16]
# fig = plot_seq_patches(seq_patches)
# plt.savefig('test.png')

class PatchEmbed(nn.Module):
    # assuming image has height=width=224
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, dilation=1):
        super().__init__()
        num_patches = (img_size//patch_size) * (img_size//patch_size)
        self.img_size = img_size 
        self.patch_size = patch_size 
        self.num_patches = num_patches 
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    # x:[num_images,channels,height,width]=[1,1,12,12]=>proj:[1,16,3,3]=>flatten:[1,16,9]=>transpose:[1,9,16]
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1,2)
        return x 

# torch.manual_seed(13)
# patch_embed = PatchEmbed(
#     img.size(-1), patch_size=kernel_size, in_channels=1, embed_dim=kernel_size**2
# )
# embedded = patch_embed(img) # [1,9,16]
# fig = plot_seq_patches(embedded[0])
# plt.savefig('test.png')

# ## Special Classifier Token [CLS]
# imgs = torch.as_tensor(images[2:4]).float() / 255.
# fig = plot_images(imgs)
# plt.savefig('test.png')
# embeddeds = patch_embed(imgs) # [2,9,16]
# fig = plot_seq_patches_transp(embeddeds, add_cls=False, title='Image / Sequence')
# plt.savefig('test.png')
# fig = plot_seq_patches_transp(embeddeds, add_cls=True, title='Image / Sequence')
# plt.savefig('test.png')
# cls_token = nn.Parameter(torch.zeros(1,1,16))
# # Fetch a batch, embed and add cls
# images, labels = next(iter(train_loader)) # images: [16,1,12,12]
# embed = patch_embed(images) # [16,9,16]
# cls_tokens = cls_token.expand(embed.size(0),-1,-1) # replicate [1,1,16]=>[16,1,16] (-1 means keeping this dim unchanged)
# embed_cls = torch.cat((cls_tokens, embed), dim=1) # [16,1,16]&[16,9,16]=>[16,10,16]

class ViT(nn.Module):
    def __init__(self, encoder, img_size, in_channels, patch_size, n_outputs):
        super().__init__()
        # Note: d_model must be patch_size*patch_size in VisionTransformer
        self.d_model = encoder.d_model 
        self.n_outputs = n_outputs 
        self.cls_token = nn.Parameter(
            torch.zeros(1,1,encoder.d_model)
        )
        self.embed = PatchEmbed(img_size, patch_size, in_channels, encoder.d_model)
        self.encoder = encoder
        self.mlp = nn.Linear(encoder.d_model, n_outputs)
        
    def preprocess(self, X):
        # [N,C,H,W]->[N,num_patches,patch_size*patch_size]
        src = self.embed(X) 
        # [1,1,D]->[N,1,D]
        cls_tokens = self.cls_token.expand(X.size(0), -1, -1)
        src = torch.cat((cls_tokens, src), dim=1) # [N,num_patches+1,patch_size**2]
        return src 
    
    def encode(self, source):
        states = self.encoder(source)
        # state from the 1st token is used to classify
        cls_state = states[:, 0] # [N,1,D]
        return cls_state 
    
    def forward(self, X):
        src = self.preprocess(X)
        cls_state = self.encode(src)
        out = self.mlp(cls_state) # [N,1,outputs]
        return out 

# torch.manual_seed(17)
# layer = EncoderLayer(n_heads=2, d_model=16, ff_units=20)
# encoder = EncoderTransf(layer, n_layers=1)
# model_vit = ViT(encoder, img_size=12, in_channels=1, patch_size=4, n_outputs=3)
# multi_loss_fn = nn.CrossEntropyLoss()
# optimizer_vit = optim.Adam(model_vit.parameters(), lr=1e-3)
# sbs_vit = StepByStep(model_vit, multi_loss_fn, optimizer_vit)
# sbs_vit.set_loaders(train_loader, val_loader)
# sbs_vit.train(20)
# fig = sbs_vit.plot_losses()
# plt.savefig('test.png')
# print("CLS token for classification:\n", model_vit.cls_token)
# print("Model Recall:\n", StepByStep.loader_apply(sbs_vit.val_loader, sbs_vit.correct))

## Putting It All Together
# Data preparation
points, directions = generate_sequences(n=256, seed=13) # [256,4,2]
full_train = torch.as_tensor(np.array(points)).float() # [256,4,2]
target_train = full_train[:, 2:] # [256,2,2]
train_data = TensorDataset(full_train, target_train)
generator = torch.Generator()
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, generator=generator)

test_points, test_directions = generate_sequences(seed=19)
full_test = torch.as_tensor(np.array(test_points)).float()
source_test = full_test[:, :2] # [256,2,2]
target_test = full_test[:, 2:] # [256,2,2]
test_data = TensorDataset(source_test, target_test)
test_loader = DataLoader(test_data, batch_size=16)

# Model configuration and training 
torch.manual_seed(42)
enclayer = EncoderLayer(n_heads=3, d_model=6, ff_units=10, dropout=0.1)
declayer = DecoderLayer(n_heads=3, d_model=6, ff_units=10, dropout=0.1)
enctransf = EncoderTransf(enclayer, n_layers=2)
dectransf = DecoderTransf(declayer, n_layers=2)
model_transf = EncoderDecoderTransf(
    enctransf,
    dectransf,
    input_len=2,
    target_len=2,
    n_features=2
)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model_transf.parameters(), lr=0.01)

for p in model_transf.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

sbs_seq_transf = StepByStep(model_transf, loss, optimizer)
sbs_seq_transf.set_loaders(train_loader, test_loader)
sbs_seq_transf.train(50)
print("Train and Val Losses: ", sbs_seq_transf.losses[-1], sbs_seq_transf.val_losses[-1])