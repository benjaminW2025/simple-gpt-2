import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GPTLoader(Dataset):
    """
    Dataset class for my mini-GPT-2
    """

    def __init__(self, filepath, seq_len):
        # first load the file input
        self.data = np.load(filepath)

        # save parameters
        self.seq_len = seq_len

        # calculate the number of sequences
        self.num_seq = len(self.data) - seq_len
        
        # check for valid sequence lengths
        if self.num_seq <= 0:
            raise ValueError("Corpus too short for the specified sequence length.")
    
    def __len__(self):
        return self.num_seq
    
    def __getitem__(self, idx):
        # create the data sequences and the targets
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]

        # convert to tensors
        x_tensor = torch.from_numpy(x.astype(np.int64)) 
        y_tensor = torch.from_numpy(y.astype(np.int64))

        return x_tensor, y_tensor