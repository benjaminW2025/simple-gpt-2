import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GPTLoader(Dataset):
    """
    Dataset class for my mini-GPT-2
    """

    def __init__(self, filepath, seq_len, stride):
        # first load the file input
        self.data = torch.load(filepath)

        # save parameters
        self.seq_len = seq_len
        self.stride = stride

        # calculate the number of sequences
        self.num_seq = (len(self.data) - seq_len) // self.stride
        
        # check for valid sequence lengths
        if self.num_seq <= 0:
            raise ValueError("Corpus too short for the specified sequence length.")
    
    def __len__(self):
        return self.num_seq
    
    def __getitem__(self, idx):
        # get starting index
        idx = idx * self.stride

        # create the data sequences and the targets
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]

        return x, y