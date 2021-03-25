# Standard lib python import
import logging
import glob
import os
import math

# Specialized python lib
import numpy as np
import gzip

# pytorch
import torch
from torch.utils.data import Subset, Dataset, DataLoader


def load_images(image_type : str, path : str="data", n_batch=4) -> np.ndarray:
	"""
	Read all the images (of certain type) located in the folder path. 
	The dtype are already encoded in float32.

	args:
		image_type (str): which image we want to get (Sinogram, FBP or Phantom)
		path (str): the path to the data files
		n_batch (int): The number of files to load (max value is 4)

	return:	
		The image dataset (ndarray)
	"""
	files = glob.glob(os.path.join(path, "{}*.npy.gz".format(image_type)))
	data = []
	logging.info(f"\n{'-' * 25}\nLoading {image_type}\n{'-' * 25}")
	for file in files[:n_batch]:
		logging.info(f"Loading file {file}")
		f = gzip.GzipFile(file, "r")
		batch = np.load(f)
		data.append(batch)

	data = np.concatenate(tuple(data))
	#print(data.shape)
	data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])
	#print(data.shape)
	#print(data[0])
	#exit(0)
	return data


def load_all_images(path : str="data", n_batch : int=4, train_split : float=0.9,  normalize : bool=False) -> tuple:
	"""
	Read all the images located in the folder path. 
	The dtype are already encoded in float32.

	args:
		path (str): the path to the data files
		n_batch (int): number of files to load
		train_split (float): fraction of data that will be used for training (train set and validation set)

	return:	
		The image dataset (dict)
	"""
	train_images = {}
	test_images = {}
	image_types = ["Sinogram", "FBP", "Phantom"]
	for image_type in image_types:
		tmp_images = load_images(image_type, n_batch=n_batch)
		if normalize:
			# Do we normalize each image individually or all at the same time?
			tmp_images -= np.min(tmp_images)
			tmp_images /= np.max(tmp_images)
		n_examples = tmp_images.shape[0]
		train_images[image_type.upper()] = tmp_images[:int(train_split * n_examples)]
		test_images[image_type.upper()] = tmp_images[int(train_split * n_examples):]
	return train_images, test_images


class BreastCTDataset(Dataset):
	def __init__(self, data : np.ndarray, targets : np.ndarray):
		# TODO
		self.data = torch.Tensor(data)
		self.targets = torch.Tensor(targets)
	
	def __len__(self):
		return len(self.targets)

	def __getitem__(self, i):
		return self.data[i], self.targets[i]

	def to_numpy(self):
		return self.data.numpy(), self.targets.numpy()
