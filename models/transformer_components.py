import math
import torch
import torch as nn

class Decoder(nn.Module):
    """
    Decoder block
    """
    def __init__(self, d_model, num_heads, d_ff, dropout, device):
        super().__init__()
        # save parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # create necessary layers
        self.dropout_one = torch.nn.Dropout(dropout)
        self.dropout_two = torch.nn.Dropout(dropout)
        self.layer_norm_one = torch.nn.LayerNorm(d_model)
        self.layer_norm_two = torch.nn.LayerNorm(d_model)
        self.self_attention = MultiHeadedAttention(d_model, num_heads, dropout, device)
        self.feed_forward = FeedForward(d_model, d_ff, torch.nn.GELU)

    def forward(self, x):
        # apply first layers
        temp = self.layer_norm_one(x)
        temp = self.self_attention(temp)
        temp = self.dropout_one(temp)

        # add residual
        x = x + temp

        # apply the rest of the layers
        temp = self.layer_norm_two(x)
        temp = self.feed_forward(temp)
        temp = self.dropout_two(temp)

        # add residual and return
        x = x + temp

        return x

class MultiHeadedAttention(nn.Module):
    """
    Class for multi-headed self attention mechanism
    """
    def __init__(self, d_model, num_heads, dropout, device):
        super().__init__()
        # store hyperparameters
        self.num_heads = num_heads
        self.d_key = d_model // self.num_heads
        self.d_model = d_model
        self.device = device

        # create the query, key, and value weight matrices
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)

        # create the final output projection matrix and dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        self.output_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        # get shape of input
        batch_size, seq_len, _ = x.shape

        # first create the query, key, and value matrices
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # need to reshape (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_key)
        Q = torch.reshape(Q, (batch_size, seq_len, self.num_heads, self.d_key))
        K = torch.reshape(K, (batch_size, seq_len, self.num_heads, self.d_key))
        V = torch.reshape(V, (batch_size, seq_len, self.num_heads, self.d_key))

        # create proper order of Q, V, K
        Q = Q.transpose(2, 1)
        K = K.transpose(2, 1)
        V = V.transpose(2, 1)

        # need to create casual mask
        casual_mask = torch.fill((seq_len, seq_len), 1)
        casual_mask = torch.triu(casual_mask, diagonal=1)
        casual_mask = casual_mask.bool().to(self.device)

        # calculate attention pattern first
        logits = (Q @ K.transpose(3, 2) / math.sqrt(self.d_key))

        # apply the casual mask
        logits = logits.masked_fill(casual_mask, float('-inf'))

        # apply softmax
        attn_scores = torch.softmax(logits, dim=-1)

        # apply drop out
        attn_scores = self.dropout(attn_scores)

        # compute output and reshape
        output = attn_scores @ V
        output = output.transpose(2, 1)
        output = output.reshape(batch_size, seq_len, self.d_model)

        # apply output linear projection matrix
        output = self.output_proj(output)

class FeedForward(nn.Module):
    """
    Simple two layer MLP
    """
    def __init__(self, d_model, d_ff, activation, dropout=0.1):
        super().__init__()
        # save parameters
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation

        # create linear layers
        self.linear_one = torch.nn.Linear(self.d_model, self.d_ff)
        self.linear_two = torch.nn.Linear(self.d_ff, self.d_model)

        # create dropout layer
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # apply transformations
        x = self.linear_one(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_two(x)

        # return result
        return x