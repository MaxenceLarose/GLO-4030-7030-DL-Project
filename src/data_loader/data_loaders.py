# Standard lib python import
import logging
import glob
import os
import math
from typing import List

# Specialized python lib
import numpy as np
import gzip


def load_images(
		image_type : str,
		path : str="data",
		n_batch=4,
		flatten_images=False,
) -> np.ndarray:
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
	files = glob.glob(os.path.join(path, "{}*.npy*".format(image_type)))
	data = []
	logging.info(f"\n{'-' * 25}\nLoading {image_type}\n{'-' * 25}")
	for file in files[:n_batch]:
		logging.info(f"Loading file {file}")
		try:
			f = gzip.GzipFile(file, "r")
			batch = np.load(f)
			f.close()
		except:
			batch = np.load(file)

		data.append(batch)
	data = np.concatenate(tuple(data))

	if flatten_images:
		data = data.reshape(data.shape[0], 1, data.shape[1] * data.shape[2])
	else:
		data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])
	return data


def load_all_images(image_types : list,
	path : str="data",
	n_batch : int=4,
	clip : bool=False,
	flatten_images : bool=False,
	images_not_to_reshape="PHANTOM",
	min_max_norm=False,
	z_norm=False) -> tuple:
	"""
	Read all the images located in the folder path.
	The dtype are already encoded in float32.

	args:s
		path (str): the path to the data files
		n_batch (int): number of files to load
		train_split (float): fraction of data that will be used for training (train set and validation set)

	return:
		The image dataset (dict)
	"""
	train_images = {}
	for image_type in image_types:
		if images_not_to_reshape == image_type:
			tmp_images = load_images(image_type, n_batch=n_batch, flatten_images=False,
									 min_max_norm=min_max_norm, z_norm=z_norm)
		else:
			tmp_images = load_images(image_type, n_batch=n_batch, flatten_images=flatten_images,
									 min_max_norm=min_max_norm, z_norm=z_norm)
		if clip:
			tmp_images = np.clip(tmp_images, 0, 1)
		n_examples = tmp_images.shape[0]

		if min_max_norm:
			if image_type == "FBP128":
				min_value, max_value = tmp_images.min(), tmp_images.max()
			logging.info(f"Initial minimum value is {min_value}\nInitial maximum value is {max_value}")
			data = (tmp_images - min_value) / (max_value - min_value)
			logging.info(f"Final minimum value is {data.min()}\nFinal maximum value is {data.max()}")

		if z_norm:
			if image_type == "FBP128":
				mean_value, std_value = np.mean(tmp_images.flatten()), np.std(tmp_images.flatten())
			logging.info(f"Initial mean value is {mean_value}\nInitial std value is {std_value}")
			data = (tmp_images - mean_value) / std_value
			logging.info(f"Final mean value is {np.mean(data.flatten())}\nFinal std value is {np.std(data.flatten())}")

		train_images[image_type.upper()] = tmp_images

	return train_images


def load_result_images(
		models: List[str] = ["InceptionUNet", "NestedUNet", "UNet"],
		image_types: List[str] = ["predictions", "targets"],
		path: str = "results",
		n_batch: int = 1,
		train_split: float = 0.9,
		clip: bool = False,
		load_sinograms=False,
		merge_datasets=False,
		ratio_of_images_to_use=1
) -> tuple:
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

	for idx, model in enumerate(models):
		model_train_images = {}
		model_test_images = {}
		for image_type in image_types:
			if image_type == "predictions":
				sub_file_name = "res"
			elif image_type == "targets":
				sub_file_name = "ref"
			else:
				raise ValueError("Image type is not allowed.")

			if idx == 0:
				tmp_images = load_images(
					image_type=f"{model}/train_images_prediction/{sub_file_name}/{image_type}",
					path=path,
					n_batch=n_batch
				)

				if clip:
					tmp_images = np.clip(tmp_images, 0, 1)

				total_examples = tmp_images.shape[0]
				number_images_to_use = int(round(total_examples * ratio_of_images_to_use))
				logging.info(f"Using {number_images_to_use} images from model {model} as {image_type}.")
				tmp_images = tmp_images[:number_images_to_use]

				n_examples = tmp_images.shape[0]
				model_train_images[f"{model}_{image_type}"] = tmp_images[:int(train_split * n_examples)]
				model_test_images[f"{model}_{image_type}"] = tmp_images[int(train_split * n_examples):]

				train_images[image_type.upper()] = model_train_images[f"{model}_{image_type}"].copy()
				test_images[image_type.upper()] = model_test_images[f"{model}_{image_type}"].copy()

			elif image_type == "predictions":
				tmp_images = load_images(
					image_type=f"{model}/train_images_prediction/{sub_file_name}/{image_type}",
					path=path,
					n_batch=n_batch
				)

				if clip:
					tmp_images = np.clip(tmp_images, 0, 1)

				total_examples = tmp_images.shape[0]
				number_images_to_use = int(round(total_examples * ratio_of_images_to_use))
				logging.info(f"Using {number_images_to_use} images from model {model} as {image_type}.")
				tmp_images = tmp_images[:number_images_to_use]

				n_examples = tmp_images.shape[0]
				model_train_images[f"{model}_{image_type}"] = tmp_images[:int(train_split * n_examples)]
				model_test_images[f"{model}_{image_type}"] = tmp_images[int(train_split * n_examples):]

				train_images[image_type.upper()] = np.concatenate(
					(train_images[image_type.upper()].copy(), model_train_images[f"{model}_{image_type}"].copy()),
					axis=1
				)

				test_images[image_type.upper()] = np.concatenate(
					(test_images[image_type.upper()].copy(), model_test_images[f"{model}_{image_type}"].copy()),
					axis=1
				)
			else:
				logging.info("\nThis model targets images are not used.")

			logging.info(f"Current shape of train images for {image_type}:{train_images[image_type.upper()].shape}.")
			logging.info(f"Current shape of test images for {image_type}:{test_images[image_type.upper()].shape}.")
	logging.info(f"\nFinal shape of train images for predictions: {train_images['PREDICTIONS'].shape}.")
	logging.info(f"Final shape of train images for targets: {test_images['TARGETS'].shape}.")
	logging.info(f"Final shape of test images for predictions: {train_images['PREDICTIONS'].shape}.")
	logging.info(f"Final shape of test images for targets: {test_images['TARGETS'].shape}.\n")

	return train_images, test_images
