import numpy as np
from typing import Tuple, List, Dict
import logging
import pprint

import poutyne as pt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class WeightedAverage(object):
    def __init__(
            self,
            loss_rmse: List[List],
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

    def evaluate_generator(self, x):
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


class EnsembleVoting(object):
    def __init__(self, method: str, input_shape: Tuple[int, int, int], loss_rmse: List[List] = None, **kwargs):
        """
        Constructor of the class VotingEnsemble.

        Args:
            method (str): Method used to calculate the result of the vote. Implemented methods are 'WeightedAverage',
                          'FCLayers', 'CNN'.
            input_shape (Tuple[int, int, int]): The input shape of the network. For example, if the size of the images
                                                are 512x512 and there is 3 models in the ensemble, the input_shape must
                                                be (3, 512, 512).
            loss_rmse (List[List]): If the chosen method is 'WeightedAverage', the value of this variable needs to be
                                    given. The loss_rmse should be given in the form of a list of loss lists. Each list
                                    contains every RMSE loss value associated with the training images. All models used
                                    in the ensemble must therefore be evaluated with the training images to obtain this
                                    list. (Default = None)

        Return:
            None
        """
        super().__init__()

        self.available_methods = [
            "WeightedAverage",
            "FCLayers",
            "CNN"
        ]

        self.loss_rmse = loss_rmse
        self.input_shape = input_shape
        self.output_shape = input_shape[1]*input_shape[2]
        self.kwargs = kwargs
        self.network = None

        self._method = method

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, method: str):
        if method not in self.available_methods:
            raise NotImplementedError(f"Chosen network isn't implemented \nImplemented networks are "
                                      f"{self.available_methods}.")

        elif method == self.available_methods[0]:
            if self.loss_rmse is None:
                raise ValueError(f"If the chosen method is {self.available_methods[0]}, the value of this variable "
                                 f"needs to be given.")
            else:
                self.network = WeightedAverage(
                    loss_rmse=self.loss_rmse,
                    input_shape=self.input_shape
                )

        elif method == self.available_methods[1]:
            self.network = FCLayersVote(
                input_shape=self.input_shape,
                output_shape=self.output_shape
            )

        elif method == self.available_methods[2]:
            self.network = CNNVote(
                input_shape=self.input_shape,
                kernel_size=self.kwargs.get("kernel_size", 1),
                padding=self.kwargs.get("padding", 0),
            )

        self._method = method

    def train_network(
            self,
            loaders: Dict[str, DataLoader],
            save_path="model/ensemble_method_model/",
            **training_kwargs
    ):
        if self._method == self.available_methods[0]:
            logging.info(f"The {self._method} doesn't require training. Use the test_network function.")
            return []
        else:
            params = [p for p in self.network.parameters() if p.requires_grad]
            if len(params) == 0:
                exec_training = False
            else:
                exec_training = True

            if exec_training:
                optimizer = torch.optim.Adam(
                    params,
                    lr=training_kwargs.get("lr", 1e-3),
                    weight_decay=training_kwargs.get("weight_decay", 1e-3),
                )
                model = pt.Model(self.network, optimizer, 'MSELoss', batch_metrics=['accuracy'])
                if torch.cuda.is_available():
                    model.cuda()

                scheduler = pt.ReduceLROnPlateau(monitor='loss', mode="min", patience=3, factor=0.5, verbose=True)

                history = model.fit_generator(
                    loaders["train"],
                    loaders["valid"],
                    epochs=training_kwargs.get("epochs", 5),
                    callbacks=[scheduler],
                )

                model.save_weights(f"{save_path}/{self._method}.pt")

                logging.info(f"history: \n{pprint.pformat(history, indent=4)}")

                return history

    def test_network(
            self,
            loaders: Dict[str, DataLoader],
            save_path="model/ensemble_method_model/"
    ) -> tuple:
        if self._method == self.available_methods[0]:
            model = self.network
            test_metrics = model.evaluate_generator(loaders["test"])

        else:
            model = pt.Model(self.network, optimizer=None, loss_function=None, batch_metrics=["accuracy"])
            model.load_weights(f"{save_path}/{self._method}.pt")
            test_metrics = model.evaluate_generator(loaders["test"])

        return test_metrics