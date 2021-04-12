# Standard lib python import
import logging
import glob
import os

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
