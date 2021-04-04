import numpy as np
import cv2

import torch
from torch.utils.data import Subset, Dataset, DataLoader


class BreastCTDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray, preprocessing=None):
        # TODO
        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets)
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):

        if self.preprocessing:
            sample = self.preprocessing(image=self.data[i].numpy(),
                                        mask=self.targets[i].numpy()
                                        )
            self.data[i], self.targets[i] = sample['image'], sample['mask']

        return self.data[i], self.targets[i]

    def to_numpy(self):
        return self.data.numpy(), self.targets.numpy()
