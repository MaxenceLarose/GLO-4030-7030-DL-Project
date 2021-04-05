import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Subset, Dataset, DataLoader


class BreastCTDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray, preprocessing=None):
        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets)
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):

        image, mask = self.data[i], self.targets[i]

        # plt.imshow(image.numpy().transpose(1, 2, 0)[:, :, 0], cmap="Greys")
        # plt.show()

        if self.preprocessing:

            sample = self.preprocessing(image=self.data[i].numpy().transpose(1, 2, 0),
                                        mask=self.targets[i].numpy().transpose(1, 2, 0)
                                        )
            image, mask = torch.from_numpy(sample['image']), torch.from_numpy(sample['mask'])

        return image, mask

    def to_numpy(self):
        return self.data.numpy(), self.targets.numpy()
