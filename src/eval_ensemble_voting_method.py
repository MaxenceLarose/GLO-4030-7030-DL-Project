import logging
from data_loader.data_loaders import load_all_images
from model.ensemble_voting_methods import EnsembleVoting
from torch.utils.data import DataLoader, Dataset
from logger.logging_tools import logs_file_setup, log_device_setup, set_seed
from deeplib.datasets import train_valid_loaders
from torchvision.transforms import ToTensor
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

    # Method
    available_methods = [
        "WeightedAverage",
        "FCLayers",
        "CNN"
    ]
    method = available_methods[1]

    # Number of networks in the ensemble
    ensemble_size = 3

    # seed
    seed = 42
    set_seed(seed)

    # --------------------------------------------------------------------------------- #
    #                                 Ensemble                                          #
    # --------------------------------------------------------------------------------- #
    ensemble_voting = EnsembleVoting(
        method=method,
        input_shape=(ensemble_size, 512, 512)
    )

    # --------------------------------------------------------------------------------- #
    #                            dataset                                                #
    # --------------------------------------------------------------------------------- #
    # README :
    # The following variable must be replaced to be defined with the datasets to test. The input of the models
    # corresponds to a matrix whose depth (number of channels) corresponds to
    # the numbers of models in the set method, while the width and height are the size of the image.
    test_dataset = Dataset()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    loaders = dict(test=test_loader)

    # --------------------------------------------------------------------------------- #
    #                          Ensemble evaluation                                      #
    # --------------------------------------------------------------------------------- #
    test_acc = ensemble_voting.test_network(
        loaders=loaders,
    )

    logging.info(f"Test accuracy: {test_acc}")
