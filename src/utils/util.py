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


def read_log_file(file_path: str, model_name: str) -> dict:
    """
    Function used to read and parse the log file containing the history.

    Args :
        file_path: File path of the log file containing the history. (str)
        model_name: Name of the model. (str)
    Returns :
        history: Dict of list containing the history of each parameter for each epoch. (dict)
    """
    with open(file_path, "r") as log_file:
        lines: List[str] = log_file.readlines()

        for line_idx, line in enumerate(lines):
            if line.__contains__("defaultdict"):
                first_index: int = line_idx
            elif line.__contains__("Nombre de paramètres"):
                logging.info(f"{model_name}")
                logging.info(f"{line}")

        history_list: List[str] = lines[first_index + 1:]
        history_str: str = textwrap.dedent(str().join(history_list))[:-2]
        history_dict: dict = ast.literal_eval(history_str)

    return history_dict


def show_learning_curve(file_paths: List[str], model_names: List[str], **kwargs) -> tuple:
    """
    Function used to plot the learning curve.

    Args :
        file_path: File paths of the log files containing the history. (List[str])
        model_names: Names of the different models. These names are the ones used in the figure. (List[str])
        kwargs: {
            save: True to save the current fig, else false. (bool)
            save_name: The save name of the current fig. Default = "figures/model.png". (str)
            show: True to show the current fig and clear the current buffer. Default = True. (bool)
            markers: List of the marker styles used for train and validation markers. (list)
            markersize: Marker size. Default = 4. (int)
            font_size: Font size for labels and legend. Default = 16. (int)
        }

    Returns :
        Fig and axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    from matplotlib import colors as mcolors

    available_markers = list(Line2D.markers.keys())[2:]
    available_colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

    total_files = len(file_paths)
    # print(available_markers)
    # markers = available_markers[:total_files]
    # markers_grouped = [markers[n:n + 2] for n in range(0, len(markers), 2)]

    markers = kwargs.get("markers", ["o", "v", "d", "x", "*", "+", "P", "s"])
    markers = markers[:total_files]

    if len(markers) != total_files:
        raise ValueError(
            f"Not enough values for markers. markers contains {len(markers)} elements while it is supposed to"
            f"contains at least {total_files} elements."
        )

    colors = available_colors[:total_files]

    fig, axes = plt.subplots(1, 1, figsize=kwargs.get("figsize", (8, 6)))

    for idx, (file_path, model_name, marker, color) in enumerate(zip(file_paths, model_names, markers, colors)):
        history: dict = read_log_file(file_path=file_path, model_name=model_name)

        epochs: list = history['epoch']

        axes.plot(
            epochs,
            history['loss'],
            marker=marker,
            markersize=kwargs.get("markersize", 4),
            linestyle='-',
            lw=1.8,
            label=f'{model_name}',
            color=color
        )

        axes.plot(
            epochs,
            history['val_loss'],
            marker=marker,
            markersize=kwargs.get("markersize", 4),
            linestyle='--',
            lw=1.8,
            color=color
        )

    fontsize = kwargs.get("font_size", 14)
    axes.set_ylabel('Loss [-]', fontsize=fontsize)
    axes.set_xlabel('Epochs [-]', fontsize=fontsize)
    axes.xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
    axes.set_yscale(kwargs.get("scale", "linear"))
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


def show_learning_curve_v2(file_paths: List[str], model_names: List[str], **kwargs) -> tuple:
    """
    Function used to plot the learning curve.

    Args :
        file_path: File paths of the log files containing the history. (List[str])
        model_names: Names of the different models. These names are the ones used in the figure. (List[str])
        kwargs: {
            save: True to save the current fig, else false. (bool)
            save_name: The save name of the current fig. Default = "figures/model.png". (str)
            show: True to show the current fig and clear the current buffer. Default = True. (bool)
            markers: List of the marker styles used for train and validation markers. (list)
            markersize: Marker size. Default = 4. (int)
            font_size: Font size for labels and legend. Default = 16. (int)
        }

    Returns :
        Fig and axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    from matplotlib import colors as mcolors

    available_markers = list(Line2D.markers.keys())[2:]
    available_colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

    total_files = len(file_paths)
    # print(available_markers)
    # markers = available_markers[:total_files]
    # markers_grouped = [markers[n:n + 2] for n in range(0, len(markers), 2)]

    markers = kwargs.get("markers", ["o", "v", "d", "x", "*", "+", "P", "s"])
    markers = markers[:total_files]

    if len(markers) != total_files:
        raise ValueError(
            f"Not enough values for markers. markers contains {len(markers)} elements while it is supposed to"
            f"contains at least {total_files} elements."
        )

    colors = available_colors[:total_files]

    fig, axes = plt.subplots(1, 1, figsize=kwargs.get("figsize", (8, 6)))

    for idx, (file_path, model_name, marker, color) in enumerate(zip(file_paths, model_names, markers, colors)):
        with open(file_path, "r") as log_file:
            lines: List[str] = log_file.readlines()

            for line_idx, line in enumerate(lines):
                if line.__contains__("epoch"):
                    first_index: int = line_idx - 1
                    break

        epochs: np.ndarray = np.loadtxt(file_path, usecols=1, skiprows=first_index)
        loss: np.ndarray = np.loadtxt(file_path, usecols=3, skiprows=first_index)
        val_loss: np.ndarray = np.loadtxt(file_path, usecols=5, skiprows=first_index)

        axes.plot(
            epochs,
            loss,
            # marker=marker,
            # markersize=kwargs.get("markersize", 4),
            linestyle='-',
            lw=1.8,
            label=f'{model_name}',
            color=color
        )

        axes.plot(
            epochs,
            val_loss,
            # marker=marker,
            # markersize=kwargs.get("markersize", 4),
            linestyle='--',
            lw=1.8,
            color=color
        )

    fontsize = kwargs.get("font_size", 14)
    axes.set_ylabel('Loss RMSE [-]', fontsize=fontsize)
    axes.set_xlabel('Epochs [-]', fontsize=fontsize)
    axes.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    axes.set_xscale(kwargs.get("scale", "linear"))
    axes.set_yscale(kwargs.get("scale", "linear"))
    axes.tick_params(axis="both", which="major", labelsize=fontsize)
    axes.legend(fontsize=fontsize)
    axes.grid()

    axes.axhline(y=0.00022, xmin=0.0001, xmax=1, lw=2, color="k")
    axes.annotate(s=r"deepx score ($2.2 \times 10^{-4}$)", xy=(1, 0.00023), fontsize=16)

    plt.tight_layout()

    if kwargs.get("save", True):
        os.makedirs("figures/", exist_ok=True)
        plt.savefig(kwargs.get("save_name", f"figures/model.png"), dpi=300)
    if kwargs.get("show", True):
        plt.show()

    return fig, axes


def show_learning_rate(file_paths: List[str], model_names: List[str], **kwargs) -> tuple:
    """
    Function used to plot the learning curve.

    Args :
        file_path: File paths of the log files containing the history. (List[str])
        model_names: Names of the different models. These names are the ones used in the figure. (List[str])
        kwargs: {
            save: True to save the current fig, else false. (bool)
            save_name: The save name of the current fig. Default = "figures/model.png". (str)
            show: True to show the current fig and clear the current buffer. Default = True. (bool)
            markers: List of the marker styles used for train and validation markers. (list)
            markersize: Marker size. Default = 4. (int)
            font_size: Font size for labels and legend. Default = 16. (int)
        }

    Returns :
        Fig and axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    from matplotlib import colors as mcolors

    available_markers = list(Line2D.markers.keys())[2:]
    available_colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

    total_files = len(file_paths)
    # print(available_markers)
    # markers = available_markers[:total_files]
    # markers_grouped = [markers[n:n + 2] for n in range(0, len(markers), 2)]

    markers = kwargs.get("markers", ["o", "v", "d", "x", "*", "+", "P", "s"])
    markers = markers[:total_files]

    if len(markers) != total_files:
        raise ValueError(
            f"Not enough values for markers. markers contains {len(markers)} elements while it is supposed to"
            f"contains at least {total_files} elements."
        )

    colors = available_colors[:total_files]

    fig, axes = plt.subplots(1, 1, figsize=kwargs.get("figsize", (8, 6)))

    for idx, (file_path, model_name, marker, color) in enumerate(zip(file_paths, model_names, markers, colors)):
        with open(file_path, "r") as log_file:
            lines: List[str] = log_file.readlines()

            for line_idx, line in enumerate(lines):
                if line.__contains__("epoch"):
                    first_index: int = line_idx - 1
                    break

        epochs: np.ndarray = np.loadtxt(file_path, usecols=1, skiprows=first_index)
        lr: np.ndarray = np.loadtxt(file_path, usecols=7, skiprows=first_index)

        axes.plot(
            epochs,
            lr,
            # marker=marker,
            # markersize=kwargs.get("markersize", 4),
            linestyle='-',
            lw=1.8,
            label=f'{model_name}',
            color=color
        )

    fontsize = kwargs.get("font_size", 14)
    axes.set_ylabel('Learning rate [-]', fontsize=fontsize)
    axes.set_xlabel('Epochs [-]', fontsize=fontsize)
    axes.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    axes.set_yscale(kwargs.get("scale", "linear"))
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
            file_paths=["train-1619896276266685.log"],
            model_names=["debug"],
            save=False,
            save_name="figures/model.png",
            show=True
        )
