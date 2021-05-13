import logging
import glob
import os
import gzip
import matplotlib.pyplot as plt

import numpy as np


def save_sparse_sinograms(
        sparse_size: int,
        path: str = "data",
        n_batch=4,
):
    """
    Save dataset of images in the folder path after transforming sinograms to sparse sinograms.

    Args:
        path (str): The path to the data files
        n_batch (int): The number of files to load (max value is 4)
        sparse_size (int): Size of the generated sparse sinogram

    Return:
        nothing
    """
    logging.info(f"\n{'-' * 25}\nGenerating SPARSE data files\n{'-' * 25}")
    for batch in range(1, n_batch + 1):
        sinogram_file = glob.glob(os.path.join(path, "Sinogram_batch{}.npy*".format(batch)))[0]
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
            sparse_sinogram = sinogram[:, ::int(sample_every_n_projection), :]
        else:
            raise ValueError(
                f"Chosen sparse size ({sparse_size}) can't divide dense sinogram size "
                f"({sinogram.shape[1]}) to whole number."
            )

        logging.info(f"Sparse sinogram size : {sparse_sinogram.shape[1]}")

        sparse_sinogram_file = "./{}/SPARSE_sinogram_batch{}.npy".format(path, batch)
        np.save(sparse_sinogram_file, sparse_sinogram)
        logging.info(f"SPARSE sinogram dataset for batch {batch} saved to file {sparse_sinogram_file}")
