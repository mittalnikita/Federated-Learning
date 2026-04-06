import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=96, pred_len=96):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return x, y


def dirichlet_split(dataset, num_clients=10, alpha=1.0, seed=42):
    np.random.seed(seed)
    n = len(dataset)
    indices = np.arange(n)
    proportions = np.random.dirichlet([alpha] * num_clients)
    sizes = (proportions * n).astype(int)
    sizes[-1] = n - sizes[:-1].sum()
    np.random.shuffle(indices)
    client_indices, start = [], 0
    for size in sizes:
        client_indices.append(indices[start: start + size])
        start += size
    return client_indices
