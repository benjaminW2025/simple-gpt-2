import torch 
from torch import nn
import numpy as np

class SinusoidalPositionalEncoding(nn.Module):
    """
    Class that handles basic sinusoidal positional encoding
    """
    def __init__(self, d_model, device, max_len=1024):
        """
        Set up and calculate the positional encodings
        
        :param d_model: depth of model
        :param max_len: maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.device = device

        # calculate the argument for the sinusoidals, first operate in log space to preserve floating point precision
        argument = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000) / d_model))

        # set up
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        self.pe = torch.zeros(max_len, d_model)

        # calculate the positional encoding for each pos and dim
        self.pe[:, 0::2] = torch.sin(position * argument)
        self.pe[:, 1::2] = torch.cos(position * argument)

        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        """
        Returns the input tensor with positional encoding
        
        :param x: input tensor
        """
        return (x + self.pe[:, :x.size(1), :].to(self.device))