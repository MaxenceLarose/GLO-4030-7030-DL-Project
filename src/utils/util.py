# Standard lib python import
import logging
import glob
import os
from typing import List
import ast
import textwrap

# Specialized python lib
import albumentations as albu
import numpy as np
import gzip


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_phantom_from_diff(fbp_data, diff_data):
    """
    Retrieve phantom images from fbp and difference (or predicted difference) data

    Args:
        fbp_data (ndarray): fbp images dataset
        diff_data (ndarray): difference between phantom and fbp datasets
    Return:
        phantom images dataset (ndarray)
    """
    return fbp_data + diff_data

def save_diff_dataset(path : str="data", n_batch=4):
    """
    Save dataset of difference between phantom and fbp datasets located in the folder path.

    args:
        path (str): the path to the data files
        n_batch (int): The number of files to load (max value is 4)

    return:
        nothing
    """
    logging.info(f"\n{'-' * 25}\nGenerating DIFF data files\n{'-' * 25}")
    for batch in range(1, n_batch + 1):
        file = glob.glob(os.path.join(path, "Phantom_batch{}.npy*".format(batch)))[0]
        logging.info(f"Loading file {file}")
        try:
            f = gzip.GzipFile(file, "r")
            phantom = np.load(f)
            f.close()
        except:
            phantom = np.load(file)
        file = glob.glob(os.path.join(path, "FBP128_batch{}.npy*".format(batch)))[0]
        logging.info(f"Loading file {file}")
        try:
            f = gzip.GzipFile(file, "r")
            fbp = np.load(f)
            f.close()
        except:
            fbp = np.load(file)
        file = "./{}/DIFF_batch{}.npy".format(path, batch)
        np.save(file, phantom - fbp)
        logging.info(f"Pĥantom and FBP difference for batch {batch} saved to file {file}")

    for batch in range(1, n_batch + 1):
        file = glob.glob(os.path.join(path, "virtual_breast_{}.npy*".format(batch)))[0]
        logging.info(f"Loading file {file}")
        try:
            f = gzip.GzipFile(file, "r")
            phantom = np.load(f)
            f.close()
        except:
            phantom = np.load(file)
        file = glob.glob(os.path.join(path, "fdk_batch{}.npy*".format(batch)))[0]
        logging.info(f"Loading file {file}")
        try:
            f = gzip.GzipFile(file, "r")
            fbp = np.load(f)
            f.close()
        except:
            fbp = np.load(file)
        file = "./{}/DATA_LEO_DIFF_batch{}.npy".format(path, batch)
        np.save(file, phantom - fbp)
        logging.info(f"Pĥantom and FBP difference for batch {batch} saved to file {file}")


def read_log_file(file_path: str) -> dict:
    """
    Function used to read and parse the log file containing the history.

    Args :
        file_path: File path of the log file containing the history. (str)

    Returns :
        history: Dict of list containing the history of each parameter for each epoch. (dict)
    """
    with open(file_path, "r") as log_file:
        lines: List[str] = log_file.readlines()

        for line_idx, line in enumerate(lines):
            if line.__contains__("defaultdict"):
                first_index: int = line_idx

        history_list: List[str] = lines[first_index + 1:]
        history_str: str = textwrap.dedent(str().join(history_list))[:-2]
        history_dict: dict = ast.literal_eval(history_str)

    return history_dict


def show_learning_curve(file_path: str, **kwargs) -> tuple:
    """
    Function used to plot the learning curve.

    Args :
        file_path: File path of the log file containing the history. (str)
        kwargs: {
            save (bool): True to save the current fig, else false.
            save_name (str): The save name of the current fig. Default = "figures/model.png"
            show (bool): True to show the current fig and clear the current buffer. Default = True.
            markers (str): Markers used for the figure. Default = 'o',
            font_size (int): Font size for labels and legend. Default = 16.
        }

    Returns :
        Fig and axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    history: dict = read_log_file(file_path=file_path)

    epoch: int = len(history['loss'])
    epochs: list = list(range(1, epoch + 1))

    fig, axes = plt.subplots(1, 1, figsize=kwargs.get("figsize", (8, 6)))

    fontsize = kwargs.get("font_size", 14)
    axes.plot(epochs, history['loss'], kwargs.get("markers", 'o'), label='Train')
    axes.plot(epochs, history['val_loss'], kwargs.get("markers", 'o'), label='Validation')
    axes.set_ylabel('Loss [-]', fontsize=fontsize)
    axes.set_xlabel('Epochs [-]', fontsize=fontsize)
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.tick_params(axis="both", which="major", labelsize=fontsize)
    axes.legend(fontsize=fontsize)
    axes.grid()

    plt.tight_layout()

    if kwargs.get("save", True):
        os.makedirs("figures/", exist_ok=True)
        plt.savefig(kwargs.get("save_name", f"figures/model.png"), dpi=300)
    if kwargs.get("show", True):
        plt.show()

    return fig, axes


if __name__ == "__main__":
    debug = True

    if debug:
        show_learning_curve(
            file_path="train-1619896276266685.log",
            save=False,
            markers="-",
            save_name="figures/model.png",
            show=True
        )
