import random
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")


class MOSIData(Dataset):
    def __init__(
        self, dataset_path, split_type="train", drop_rate=0.6, full_data=False
    ):
        super(MOSIData, self).__init__()
        dataset = pickle.load(open(dataset_path, "rb"))
        self.vision = (
            torch.tensor(dataset[split_type]["vision"].astype(np.float32))
            .cpu()
            .detach()
        )
        self.text = (
            torch.tensor(dataset[split_type]["text"].astype(np.float32)).cpu().detach()
        )
        self.audio = dataset[split_type]["audio"].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = (
            torch.tensor(dataset[split_type]["labels"].astype(np.float32))
            .cpu()
            .detach()
        )

        self.drop_rate = drop_rate
        self.full_data = full_data
        self.n_modalities = 3  # vision/ text/ audio

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_lbl_info(self):
        return self.labels.shape[1], self.labels.shape[2]

    def get_missing_mode(self):
        if self.full_data:
            return 6
        if random.random() < self.drop_rate:
            return random.randint(0, 5)
        else:
            return 6

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        return X, Y, self.get_missing_mode()
