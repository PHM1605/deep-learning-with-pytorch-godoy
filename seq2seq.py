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

class EncoderDecoderSelfAttn(nn.Module):
    def __init__(self, encoder, decoder, input_len, target_len):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        self.input_len = input_len 
        self.target_len = target_len 
        self.trg_masks = subsequent_mask(self.target_len)

    def encode(self, source_seq, source_mask):
        encoder_states = self.encoder(source_seq, source_mask)
        self.decoder.init_keys(encoder_states)
    
    # use in TRAIN mode 
    def decode(self, shifted_target_seq, source_mask=None, target_mask=None):
        outputs = self.decoder(shifted_target_seq, source_mask=source_mask, target_mask=target_mask)
        return outputs 
    
    # use in VAL mode
    def predict(self, source_seq, source_mask):
        inputs = source_seq[:, -1:]
        for i in range(self.target_len):
            out = self.decode(inputs, source_mask, self.trg_masks[:,:i+1, :i+1])
            out = torch.cat([inputs, out[:,-1:,:]], dim=-2)
            inputs = out.detach()
        outputs = inputs[:, 1:, :]
        return outputs

    def forward(self, X, source_mask=None):
        self.trg_masks = self.trg_masks.type_as(X).bool()
        source_seq = X[:, :self.input_len, :]
        self.encode(source_seq, source_mask)
        if self.training:
            shifted_target_seq = X[:, self.input_len-1:-1, :]
            outputs = self.decode(shifted_target_seq, source_mask, self.trg_masks)
        else:
            outputs = self.predict(source_seq, source_mask)
        return outputs

class SubLayerWrapper(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__() 
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, sublayer, is_self_attn=False, **kwargs):
        norm_x = self.norm(x) 
        if is_self_attn:
            sublayer.init_keys(norm_x)
        out = x + self.drop(sublayer(norm_x, **kwargs))
        return out 

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads 
        self.d_model = d_model 
        self.ff_units = ff_units 
        self.self_attn_heads = MultiHeadedAttention(
            n_heads, d_model, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, d_model)
        )
        self.sublayers = nn.ModuleList([
            SubLayerWrapper(d_model, dropout)
            for _ in range(2)
        ])
    
    def forward(self, query, mask=None):
        # SubLayer 0 - Self-Attention
        att = self.sublayers[0](
            query,
            sublayer = self.self_attn_heads,
            is_self_attn=True,
            mask=mask  
        )
        # SubLayer 1 - FFN
        out = self.sublayers[1](
            att, sublayer=self.ffn
        )
        return out 

class EncoderTransf(nn.Module):
    def __init__(self, encoder_layer, n_layers=1, max_len=100):
        super().__init__()
        self.d_model = encoder.d_model 
        self.pe = PositionalEncoding(max_len, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.layers = nn.Modulelist([
            copy.deepcopy(encoder_layer)
            for _ in range(n_layers)
        ])
    
    def forward(self, query, mask=None):
        x = self.pe(query)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
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
        self.sublayers = nn.ModuleList(
            [SubLayerWrapper(d_model, dropout) for _ in range(3)]
        )

    def init_keys(self, states):
        self.cross_attn_heads.init_keys(states)
    
    def forward(self, query, source_mask=None, target_mask=None):
        # SubLayer 0 - Masked Self-Attention 
        att1 = self.sublayers[0](
            query, mask=target_mask, sublayer=self.self_attn_heads, is_self_attn=True
        )
        # SubLayer 1 - Cross-Attention
        att2 = self.sublayers[1](
            att1, mask=source_mask, sublayer=self.cross_attn_heads
        )
        # SubLayer 2 - FFN
        out = self.sublayers[2](
            att2, sublayer=self.ffn 
        )
        return out 

class DecoderTransf(nn.Module):
    def __init__(self, decoder_layer, n_layers=1, max_len=100):
        super(DecoderTransf, self).__init__()
        self.d_model = decoder_layer.d_model 
        self.pe = PositionalEncoding(max_len, self.d_model)
        self.norm = LayerNorm(self.d_model)
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

class MultiHeadedAttention(nn.Module):
    # Notice: for self-attention d_model = L
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
        # x: [N,L,D]=[N,d_model,d_model] for self-attention
        batch_size, seq_len = x.size(0), x.size(1)
        # x: [N,L,n_heads,d_head] with L=d_model=n_heads*d_head
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        # x: [N,n_heads,L,d_head]
        x = x.transpose(1,2)
        return x 
    
    def init_keys(self, key):
        # [N,n_heads,L,d_head]
        self.proj_key = self.make_chunks(self.linear_key(key))
        self.proj_value = self.make_chunks(self.linear_value(key))

    def score_function(self, query):
        # query: [N,L,L]-> proj_query: [N,n_heads,L,d_head]
        proj_query = self.make_chunks(self.linear_query(query))
        # [N,n_heads,L,d_head]x[N,n_heads,d_head,L]->[N,n_heads,L,L]
        dot_products = torch.matmul(
            proj_query, self.proj_key.transpose(-2,-1)
        )
        scores = dot_products / np.sqrt(self.d_k)
        return scores

    def attn(self, query, mask=None):
        # query: [N,L,D]->scores: [N,n_heads,L,L]
        scores = self.score_function(query)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        alphas = F.softmax(scores, dim=-1)
        alphas = self.dropout(alphas)
        self.alphas = alphas.detach() # [N,n_heads,L,L]
        # [N,n_heads,L,L]x[N,n_heads,L,d_head]->[N,n_heads,L,d_head]
        context = torch.matmul(alphas, self.proj_value)
        return context

    def output_function(self, contexts):
        out = self.linear_out(contexts)
        return out 
    
    def forward(self, query, mask=None):
        if mask is not None:
            # [N,L,L]->[N,1,L,L] (every head uses the same mask)
            mask = unsqueeze(1)        
        context = self.attn(query, mask=mask)
        context = context.transpose(1,2).contiguous() # [N,L,n_heads,d_head]
        context = context.view(query.size(0), -1, self.d_model) # [N,L,d_model]
        out = self.output_function(context) # [N,L,d_model]
        return out 