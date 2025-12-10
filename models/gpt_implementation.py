import torch
import torch.nn as nn
from transformer_components import Decoder
from positional_encodings import SinusoidalPositionalEncoding

class GPT2Model(nn.Module):
    """
    Class for GPT2 implementation
    """
    def __init__(self, vocab_size, n_layers, d_model, num_heads, d_ff, dropout, device, max_len=1024):
        super().__init__()
        # map each token id to a d_model dimension vector space
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # create the positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)

        # create dropout layer for after embedding
        self.dropout = nn.Dropout(dropout)

        # create the decoder layers
        self.layers = nn.ModuleList([Decoder(d_model, num_heads, d_ff, dropout, device) 
                       for _ in range(n_layers)])
        
        # create the layer norm
        self.final_norm = nn.LayerNorm(d_model)

