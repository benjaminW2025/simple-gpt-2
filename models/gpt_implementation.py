from torch import nn
from models.transformer_components import Decoder
from models.positional_encodings import SinusoidalPositionalEncoding

class GPT2Model(nn.Module):
    """
    Class for GPT2 implementation
    """
    def __init__(self, vocab_size, n_layers, d_model, num_heads, d_ff, activation, dropout, device, max_len=1024):
        super().__init__()
        # map each token id to a d_model dimension vector space
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # create the positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, device, max_len)

        # create dropout layer for after embedding
        self.dropout = nn.Dropout(dropout)

        # create the decoder layers
        self.layers = nn.ModuleList([Decoder(d_model, num_heads, d_ff, activation, dropout, device) 
                       for _ in range(n_layers)])
        
        # create the layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # create the final projection layer
        self.final_proj = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()
    
    def _init_weights(self):
        # explicit initialization of the weights in order to break symmetry and stablize gradients
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, targets=None):
        batch, seq_len = x.size()

        # embed the input
        embedding = self.token_embeddings(x)

        # calculate positional encoding and apply dropout
        x = self.positional_encoding(embedding)
        x = self.dropout(x)

        # apply decoder layers
        for layers in self.layers:
            x = layers(x)
        
        # apply final layer norm
        x = self.final_norm(x)

        # project into vocab space
        logits = self.final_proj(x)

        # now we calculate cross entropy loss
        loss = None
        if targets is not None:
            # reshape tensor
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
        # return
        return logits, loss

