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