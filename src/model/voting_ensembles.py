import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict


class WeightedAverage(object):
    def __init__(
            self,
            loss_rmse: List[List, ..., List],
            input_shape: Tuple[int, int, int],
    ):
        super().__init__()

        if len(loss_rmse) != input_shape[0]:
            raise ValueError("Each channel of the input matrix should be associated with an RMSE loss. The loss needs"
                             "to be calculated and given as a list of lists of losses to this class.")

        self.loss_rmse = loss_rmse
        self.weights = np.zeros(input_shape[0])
        self.conv1 = nn.Conv2d(input_shape[0], 1, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU()

    def get_model_specific_weighted(self):
        weights = np.average(np.array(self.loss_rmse))

        return weights

    def eval(self, x):
        self.weights = self.get_model_specific_weighted()
        self.conv1.weight = torch.nn.Parameter(torch.from_numpy(self.weights))
        out = self.conv1(x)
        out = self.relu1(out)

        return out


class FCLayersVote(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            output_shape: int
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(int(np.prod(input_shape)), output_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.flatten(x)
        out = self.linear(out)
        out = self.relu(out)

        return out


class CNNVote(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            kernel_size: int = 1,
            padding: int = 0
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 1, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        return out


class VotingEnsemble(object):
    def __init__(self, method: str):
        """
        Constructor of the class VotingEnsemble.

        Args:
            method (str): Method used to calculate the result of the vote. Implemented methods are 'WeightedAverage',
                          'FCLayers', 'CNN'.
        Return:
            None
        """
        super().__init__()
        available_methods = [
            "WeightedAverage",
            "FCLayers",
            "CNN"
        ]

        if method not in available_methods:
            raise NotImplementedError(f"Chosen network isn't implemented \nImplemented networks are "
                                      f"{available_methods}.")
        else:
            self.method = method

