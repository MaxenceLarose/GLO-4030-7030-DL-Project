import numpy as np

import torch
from torch.utils.data import Subset, Dataset, DataLoader


class BreastCTDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray):
        # TODO
        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]

    def to_numpy(self):
        return self.data.numpy(), self.targets.numpy()
