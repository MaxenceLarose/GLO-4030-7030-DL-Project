import numpy as np
import os
from typing import Tuple, List, Dict
import logging
from typing import Callable

import poutyne as pt
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
from .metrics import RMSELoss
from sklearn.metrics import mean_squared_error
from .inceptionNet import InceptionBlock


def initialize_network_(network: nn.Module, initialization_function_: Callable, **func_kwargs):
    """
    Function used to initialize the weights of a neural network with the given initialization function.
    The bias weights will be initialized to zero.

    Args :
        network: The neural network that will be initialized.
        initialization_function_: The initialization function. A callable that take weights as a torch.Tensor and
                                  other kwargs. The modification must be done inplace.
        func_kwargs: The kwargs of the initialization function.

    Returns :
        None
    """
    for module in network.modules():
        if isinstance(module, nn.Conv2d):
            initialization_function_(module.weight, **func_kwargs)
            init.zeros_(module.bias)


class WeightedAverage(nn.Module):
    def __init__(
            self,
            loss_rmse: List[float],
            input_shape: Tuple[int, int, int],
    ):
        super().__init__()

        if len(loss_rmse) != input_shape[0]:
            raise ValueError("Each channel of the input matrix should be associated with an RMSE loss. The loss needs"
                             "to be calculated and given as a list of lists of losses to this class.")

        self.loss_rmse = loss_rmse

        self.weights = np.zeros((1, input_shape[0], 1, 1))
        self.update_weights()

        self.conv1 = nn.Conv2d(input_shape[0], 1, kernel_size=1, padding=0)

        self.conv1.weight = torch.nn.Parameter(torch.tensor(self.weights, dtype=torch.float32), requires_grad=False)
        self.conv1.bias.data = torch.tensor(np.array([0.0]), dtype=torch.float32)
        self.conv1.bias.requires_grad = False

        self.relu1 = nn.ReLU()

    def update_weights(self) -> None:
        total_loss = np.sum(np.array(self.loss_rmse))

        for idx, loss in enumerate(self.loss_rmse):
            self.weights[0, idx, 0, 0] = - loss/total_loss + 2/len(self.loss_rmse)

        # -- Debug -- #
        # self.weights[0, 0, 0, 0] = 1
        # self.weights[0, 1, 0, 0] = 0
        # self.weights[0, 2, 0, 0] = 0

    def forward(self, x):
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
        self.inceptionBlock = InceptionBlock(input_shape[0], 1, batch_norm_momentum=0.01)
        self.conv1 = nn.Conv2d(4, 1, kernel_size=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.inceptionBlock(x)
        out = self.conv1(out)
        out = self.relu(out)

        return out


class EnsembleVoting(object):
    def __init__(self, method: str, input_shape: Tuple[int, int, int], loss_rmse: List[float] = None, **kwargs):
        """
        Constructor of the class VotingEnsemble.

        Args:
            method (str): Method used to calculate the result of the vote. Implemented methods are 'WeightedAverage',
                          'FCLayers', 'CNN'.
            input_shape (Tuple[int, int, int]): The input shape of the network. For example, if the size of the images
                                                are 512x512 and there is 3 models in the ensemble, the input_shape must
                                                be (3, 512, 512).
            loss_rmse (List[float]): If the chosen method is 'WeightedAverage', the value of this variable needs to be
                                    given. The loss_rmse should be given in the form of a list of loss. The loss is
                                    given as an RMSE loss value associated with the training images. All models used
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

        self.method = method

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
            save_path="model/ensemble_method_models_weights/",
            **training_kwargs
    ) -> List[Dict]:
        if self._method == self.available_methods[0]:
            logging.info(f"\nThe {self._method} doesn't require training. Use the test_network function.")
            return []
        else:

            init_funcs = {
                "Constant": dict(func=init.constant_, func_kwargs=dict(val=1/self.input_shape[0])),
                "Xavier_Normal": dict(func=init.xavier_normal_, func_kwargs=dict(gain=1)),
                "Kaiming_Uniform": dict(func=init.kaiming_normal_, func_kwargs=dict(a=1)),
            }

            params = init_funcs[training_kwargs.get("initialization", "Constant")]
            initialize_network_(self.network, params["func"], **params["func_kwargs"])
            params = [p for p in self.network.parameters() if p.requires_grad]

            if len(params) == 0:
                exec_training = False
            else:
                exec_training = True

            if exec_training:
                optimizer = torch.optim.Adam(
                    params,
                    lr=training_kwargs.get("lr", 1e-3),
                    weight_decay=training_kwargs.get("weight_decay", 1e-2),
                )

                model = pt.Model(self.network, optimizer, RMSELoss())
                if torch.cuda.is_available():
                    model.cuda()

                scheduler = pt.ReduceLROnPlateau(monitor='loss', mode="min", patience=3, factor=0.5, verbose=True)

                history = model.fit_generator(
                    loaders["train"],
                    loaders["valid"],
                    epochs=training_kwargs.get("epochs", 5),
                    callbacks=[scheduler],
                )

                logging.info(f"\nModel final weights are \n{model.get_weights()}.")

                folder_path = os.path.join(os.getcwd(), save_path)
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)

                model.save_weights(f"{save_path}/{self._method}.pt")

                return history

    def test_network(
            self,
            loaders: Dict[str, DataLoader],
            save_path="model/ensemble_method_models_weights/"
    ) -> tuple:
        if self._method == self.available_methods[0]:
            model = pt.Model(self.network, optimizer=None, loss_function=RMSELoss())
        else:
            model = pt.Model(self.network, optimizer=None, loss_function=RMSELoss())
            model.load_weights(f"{save_path}/{self._method}.pt")

        logging.info(f"\nModel weights are \n{model.get_weights()}.\n")

        test_metrics = model.evaluate_generator(loaders["test"])

        # -- Debug -- #
        # test_metrics = model.evaluate_generator(loaders["test"], return_pred=True, return_ground_truth=True)
        # print(np.sqrt(mean_squared_error(test_metrics[1][0][0, :, :], test_metrics[2][0][0, :, :])))
        # test_metrics = test_metrics[0]

        return test_metrics
