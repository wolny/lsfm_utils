import torch
from torch.utils.data import Dataset


class Random3DDataset(Dataset):
    def __init__(self, N, size, out_channels):
        raw_dims = (N, 1) + size
        labels_dims = (N, out_channels) + size
        self.raw = torch.randn(raw_dims)
        self.labels = torch.empty(labels_dims, dtype=torch.float).random_(2)

    def __len__(self):
        return self.raw.size(0)

    def __getitem__(self, idx):
        return self.raw[idx], self.labels[idx]
