import logging
import glob
import os
import gzip
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import interp1d


def save_sparse_sinograms(
        sparse_size: int,
        path: str = "data",
        filename: str = "Sinogram",
        n_batch=4,
):
    """
    Save dataset of images in the folder path after transforming sinograms to sparse sinograms.

    Args:
        sparse_size (int): Size of the generated sparse sinogram
        path (str): The path to the data files
        n_batch (int): The number of files to load (max value is 4)

    Return:
        nothing
    """
    logging.info(f"\n{'-' * 25}\nGenerating SPARSE data files\n{'-' * 25}")
    for batch in range(1, n_batch + 1):
        sinogram_file = glob.glob(os.path.join(path, "{}_batch{}.npy*".format(filename,batch)))[0]
        logging.info(f"Loading file {sinogram_file}")
        try:
            f = gzip.GzipFile(sinogram_file, "r")
            sinogram = np.load(f)
            f.close()
        except:
            sinogram = np.load(sinogram_file)

        sinogram_size = sinogram.shape[1]
        logging.info(f"Dense sinogram size : {sinogram_size}")

        sample_every_n_projection = sinogram_size / sparse_size

        if np.ceil(sample_every_n_projection) == sample_every_n_projection:
            sparse_sinograms = sinogram[:, ::int(sample_every_n_projection), :]
        else:
            raise ValueError(
                f"Chosen sparse size ({sparse_size}) can't divide dense sinogram size "
                f"({sinogram.shape[1]}) to whole number."
            )

        logging.info(f"Sparse sinogram size : {sparse_sinograms.shape[1]}")

        sparse_sinogram_file = "./{}/SPARSE_sinogram_batch{}.npy".format(path, batch)
        np.save(sparse_sinogram_file, sparse_sinograms)
        logging.info(f"SPARSE sinogram dataset for batch {batch} saved to file {sparse_sinogram_file}")


def interpolation_function(
        sinogram_column: np.ndarray,
        interpolation_coefficient: int,
) -> np.ndarray:
    """
    Interpolation function used to interpolate dense sinograms from sparse sinograms.

    Args:
        sinogram_column (np.ndarray): Size of the generated interpolated sinogram.
        interpolation_coefficient (int): This number specifies by how many times the number of projections should be
                                         increased.

    Return:
        New row for the interpolated sinograms.
    """
    length_new_x = int(len(sinogram_column)*interpolation_coefficient)

    x = list(range(length_new_x)[::int(interpolation_coefficient)])
    x_new = list(range(length_new_x))

    func = interp1d(x, sinogram_column, kind="linear", fill_value="extrapolate")

    return func(x_new)


def save_interpolated_sinograms(
        interpolated_size: int,
        path: str = "data",
        n_batch=4,
        show: bool = False
):
    """
    Save dataset of images in the folder path after transforming sinograms to sparse sinograms.

    Args:
        interpolated_size (int): Size of the generated interpolated sinogram
        path (str): The path to the data files
        n_batch (int): The number of files to load (max value is 4)
        show (bool): Show random sparse and interpolated sinogram.

    Return:
        nothing
    """
    logging.info(f"\n{'-' * 25}\nGenerating INTER data files\n{'-' * 25}")
    for batch in range(1, n_batch + 1):
        sparse_sinogram_file = glob.glob(os.path.join(path, "SPARSE_sinogram_batch{}.npy*".format(batch)))[0]
        logging.info(f"Loading file {sparse_sinogram_file}")
        try:
            f = gzip.GzipFile(sparse_sinogram_file, "r")
            sparse_sinograms = np.load(f)
            f.close()
        except:
            sparse_sinograms = np.load(sparse_sinogram_file)

        sparse_sinogram_size = sparse_sinograms.shape[1]
        logging.info(f"Dense sinogram size : {sparse_sinogram_size}")

        interpolated_n_times_more_projections = interpolated_size / sparse_sinogram_size

        if np.ceil(interpolated_n_times_more_projections) == interpolated_n_times_more_projections:
            interpolated_sinogram = np.apply_along_axis(
                interpolation_function,
                axis=1,
                arr=sparse_sinograms,
                interpolation_coefficient=interpolated_n_times_more_projections
            )

        else:
            raise ValueError(
                f"Chosen interpolated size ({interpolated_size}) can't be divide by sparse sinogram size "
                f"({sparse_sinograms.shape[1]}) to whole number."
            )

        logging.info(f"Interpolated sinogram size : {interpolated_sinogram.shape[1]}")

        interpolated_sinogram_file = "./{}/INTER_sinogram_batch{}.npy".format(path, batch)
        np.save(interpolated_sinogram_file, interpolated_sinogram)
        logging.info(f"INTER sinogram dataset for batch {batch} saved to file {interpolated_sinogram_file}")

        if show:
            image_idx = 50

            fig, ax = plt.subplots(2, 1, figsize=(8, 14))
            ax[0].imshow(sparse_sinograms[image_idx, :, :], cmap="Greys")
            ax[0].set_title("SPARSE")
            ax[1].imshow(interpolated_sinogram[image_idx, :, :], cmap="Greys")
            ax[1].set_title("INTERPOLATED")
            plt.show()
