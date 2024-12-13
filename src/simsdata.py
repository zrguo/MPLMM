import torch
from torch.utils.data import Dataset
import pickle
import random


class SIMSData(Dataset):
    def __init__(self, data_path, split, drop_rate, full_data=False):
        super(SIMSData, self).__init__()
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        self.data = data[split]
        self.split = split
        self.drop_rate = drop_rate
        self.full_data = full_data
        self.orig_dims = [
            self.data['text'][0].shape[1],
            self.data['audio'][0].shape[1],
            self.data['vision'][0].shape[1]
        ]

    def get_dim(self):
        return self.orig_dims
    
    def get_seq_len(self):
        return self.data['text'][0].shape[0], self.data['audio'][0].shape[0], self.data['vision'][0].shape[0]

    def __len__(self):
        return self.data['audio'].shape[0]
    
    def get_missing_mode(self):
        if self.full_data:
            return 6
        if random.random() < self.drop_rate:
            return random.randint(0, 5) 
        else:
            return 6


    def __getitem__(self, idx):
        L_feat = torch.tensor(self.data['text'][idx]).float()
        A_feat = torch.tensor(self.data['audio'][idx]).float()
        V_feat = torch.tensor(self.data['vision'][idx]).float()
        label = torch.tensor(self.data['regression_labels'][idx]).float()
        X = (L_feat, A_feat, V_feat)
        missing_code = self.get_missing_mode()

        return X, label, missing_code