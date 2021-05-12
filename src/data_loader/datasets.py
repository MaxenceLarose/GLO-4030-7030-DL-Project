import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

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

def train_valid_loaders(dataset, batch_size, train_split=0.8, shuffle=True, seed=42):
    """
    Divise un jeu de données en ensemble d'entraînement et de validation et retourne pour chacun un DataLoader PyTorch.

    Args:
        dataset (torch.utils.data.Dataset): Un jeu de données PyTorch
        batch_size (int): La taille de batch désirée pour le DataLoader
        train_split (float): Un nombre entre 0 et 1 correspondant à la proportion d'exemple de l'ensemble
            d'entraînement.
        shuffle (bool): Si les exemples sont mélangés aléatoirement avant de diviser le jeu de données.
        seed (int): Le seed aléatoire pour que l'ordre des exemples mélangés soit toujours le même.

    Returns:
        Tuple (DataLoader d'entraînement, DataLoader de test).
    """
    num_data = len(dataset)
    indices = np.arange(num_data)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    if train_split < 1 and train_split >= 0:
        split = math.floor(train_split * num_data)
        train_idx, valid_idx = indices[:split], indices[split:]

        train_dataset = Subset(dataset, train_idx)
        valid_dataset = Subset(dataset, valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    elif train_split == 1:
        dataset = Subset(dataset, indices)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        valid_loader = None
    else:
        raise RuntimeError("train_valid_loaders wrong train split input. \
            Excpected value between 0 and 1 but got {} instead".format(train_split))
    return train_loader, valid_loader
