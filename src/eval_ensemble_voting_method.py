import logging
import numpy as np

from data_loader.data_loaders import load_all_images
from model.ensemble_voting_methods import EnsembleVoting
from torch.utils.data import DataLoader, Dataset
from logger.logging_tools import logs_file_setup, log_device_setup, set_seed
from deeplib.datasets import train_valid_loaders
from torchvision.transforms import ToTensor
from data_loader.data_loaders import load_result_images
from data_loader.datasets import BreastCTDataset

"""
Eval with Voting ensemble method.
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
    # dataset constants
    batch_size = 1

    # Models
    models = ["InceptionUNet", "NestedUNet", "UNet", "BreastUNet", "Pretrained RED_CNN"]

    # Method
    available_methods = [
        "WeightedAverage",
        "FCLayers",
        "CNN"
    ]
    method = available_methods[0]

    # Number of networks in the ensemble
    ensemble_size = 5

    criterion = "RMSELoss"

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
        loss_rmse=models_loss
    )

    # --------------------------------------------------------------------------------- #
    #                            dataset                                                #
    # --------------------------------------------------------------------------------- #
    # README :
    # The following variable must be replaced to be defined with the datasets to test. The input of the models
    # corresponds to a matrix whose depth (number of channels) corresponds to
    # the numbers of models in the set method, while the width and height are the size of the image.
    _train_images, test_images = load_result_images(
        models=models,
        image_types=["predictions", "targets"],
        n_batch=1,
        ratio_of_images_to_use=1
    )

    test_dataset = BreastCTDataset(
        test_images["PREDICTIONS"],
        test_images["TARGETS"]
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    loaders = dict(test=test_loader)

    # --------------------------------------------------------------------------------- #
    #                          Ensemble evaluation                                      #
    # --------------------------------------------------------------------------------- #
    test_acc = ensemble_voting.test_network(
        loaders=loaders,
        criterion=criterion
    )

    logging.info(f"Loss ({criterion}): {test_acc}")
