import logging
import pprint
import numpy as np

from data_loader.data_loaders import load_all_images
from model.ensemble_voting_methods import EnsembleVoting
from torch.utils.data import DataLoader, Dataset
from logger.logging_tools import logs_file_setup, log_device_setup, set_seed
from torchvision.transforms import ToTensor
from data_loader.data_loaders import load_result_images
from data_loader.datasets import BreastCTDataset, train_valid_loaders

"""
Train with Voting ensemble method. You need to have multiple models already trained and saved in the repository.
"""

if __name__ == '__main__':
    # --------------------------------------------------------------------------------- #
    #                            Logs Setup                                             #
    # --------------------------------------------------------------------------------- #
    logs_file_setup(__file__, logging.INFO)
    log_device_setup()

    # --------------------------------------------------------------------------------- #
    #                            Constants                                              #
    # --------------------------------------------------------------------------------- #
    # training setup constants
    lr = 0.01
    epochs = 100
    weight_decay = 1e-4

    # dataset constants
    batch_size = 1

    # Models
    models = ["BreastUNet", "InceptionUNet", "UNet", "NestedUNet"]

    # Methods
    available_methods = [
        "WeightedAverage",
        "FCLayers",
        "CNN"
    ]
    method = available_methods[2]

    # Number of networks in the ensemble
    ensemble_size = 4

    # seed
    seed = 42
    set_seed(seed)

    # --------------------------------------------------------------------------------- #
    #                              Models scores                                        #
    # --------------------------------------------------------------------------------- #
    models_loss: list = [
        np.loadtxt(fname=f"results/{model}/train_images_prediction/scores.txt", usecols=1)[0] for model in models
    ]

    # --------------------------------------------------------------------------------- #
    #                                 Ensemble                                          #
    # --------------------------------------------------------------------------------- #
    ensemble_voting = EnsembleVoting(
        method=method,
        input_shape=(ensemble_size, 512, 512),
        loss_rmse=models_loss,
        kernel_size=3,
        padding=1
    )

    # --------------------------------------------------------------------------------- #
    #                            dataset                                                #
    # --------------------------------------------------------------------------------- #
    # The input of the models corresponds to a matrix whose depth (number of channels) corresponds to
    # the numbers of models in the set method, while the width and height are the size of the image.

    train_images, _test_images = load_result_images(
        models=models,
        image_types=["predictions", "targets"],
        n_batch=1,
        ratio_of_images_to_use=1
    )

    train_valid_dataset = BreastCTDataset(
        train_images["PREDICTIONS"],
        train_images["TARGETS"]
    )

    train_loader, valid_loader = train_valid_loaders(train_valid_dataset, batch_size=batch_size, train_split=0.9)
    loaders = dict(train=train_loader, valid=valid_loader)

    # --------------------------------------------------------------------------------- #
    #                          Ensemble training                                        #
    # --------------------------------------------------------------------------------- #

    history = ensemble_voting.train_network(
        loaders=loaders,
        initialization="Xavier_Normal",
        lr=lr,
        epochs=epochs,
        weight_decay=weight_decay
    )

    # --------------------------------------------------------------------------------- #
    #                           network analysing                                       #
    # --------------------------------------------------------------------------------- #
    logging.info(f"history: \n{pprint.pformat(history, indent=4)}")
