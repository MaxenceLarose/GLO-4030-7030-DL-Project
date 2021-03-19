# Standard lib python import
import glob
import os

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
	for file in files[:n_batch]:
		f = gzip.GzipFile(file, "r")
		batch = np.load(f)
		data.append(batch)
	data = np.concatenate(tuple(data))
	return data

def load_all_images(path : str="data", n_batch : int=4, train_split : float=0.9) -> tuple:
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
		n_examples = tmp_images.shape[0]
		train_images[image_type.upper()] = tmp_images[:int(train_split * n_examples)]
		test_images[image_type.upper()] = tmp_images[int(train_split * n_examples):]
	return train_images, test_images

def train_valid_loaders(dataset : Dataset, batch_size : int, train_split : float=0.8, shuffle : bool=True, seed : int=42):
	"""
	Divise un jeu de données en ensemble d'entraînement et de validation et retourne pour chacun un DataLoader PyTorch.

	Args:
		dataset (torch.utils.data.Dataset): Un jeu de données PyTorch
		batch_size (int): La taille de batch désirée pour le DataLoader
		train_split (float): Un nombre entre 0 et 1 correspondant à la proportion d'exemple de l'ensemble
			d'entraînement.
		shuffle (bool): Si les exemples sont mélangés aléatoirement avant de diviser le jeu de données.
		seed (int): Le seed aléatoire pour que l'ordre des exemples mélangés soit toujours le même.

	Returns:
		Tuple (DataLoader d'entraînement, DataLoader de test).
	"""
	num_data = len(dataset)
	indices = np.arange(num_data)

	if shuffle:
		np.random.seed(seed)
		np.random.shuffle(indices)

	split = math.floor(train_split * num_data)
	train_idx, valid_idx = indices[:split], indices[split:]

	train_dataset = Subset(dataset, train_idx)
	valid_dataset = Subset(dataset, valid_idx)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

	return train_loader, valid_loader


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
