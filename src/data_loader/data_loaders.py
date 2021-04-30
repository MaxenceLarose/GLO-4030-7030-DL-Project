# Standard lib python import
import logging
import glob
import os
import math

# Specialized python lib
import numpy as np
import gzip


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
	data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])
	print(np.where(data > 1e10))
	return data


def load_all_images(image_types : list=["Sinogram", "FBP", "Phantom"],
	path : str="data",
	n_batch : int=4,
	train_split : float=0.9,
	clip : bool=False,
	load_sinograms=False,
	multiple_channels=False,
	merge_datasets=False) -> tuple:
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
	
	#np.random.seed(42)
	for image_type in image_types:
		if not load_sinograms and image_type == "Sinogram":
			continue
		tmp_images = load_images(image_type, n_batch=n_batch)
		if clip:
			tmp_images = np.clip(tmp_images, 0, 1)
		n_examples = tmp_images.shape[0]
		train_images[image_type.upper()] = tmp_images[:int(train_split * n_examples)]
		test_images[image_type.upper()] = tmp_images[int(train_split * n_examples):]

	if multiple_channels and "RECLEO" in train_images.keys():
		train_images["FBP"] = np.concatenate((train_images["FBP"], train_images["RECLEO"]), axis=1)
		test_images["FBP"] = np.concatenate((test_images["FBP"], test_images["RECLEO"]), axis=1)
	if "RECLEO" in train_images.keys() and not multiple_channels:
		if merge_datasets:
			train_images["FBP"] = np.concatenate((train_images["FBP"], train_images["RECLEO"]), axis=0)
			test_images["FBP"] = np.concatenate((test_images["FBP"], test_images["RECLEO"]), axis=0)
	if "VIRTUAL_BREAST" in train_images.keys():
		try:
			if merge_datasets:
				train_images["PHANTOM"] = np.concatenate((train_images["PHANTOM"], train_images["VIRTUAL_BREAST"]), axis=0)
				test_images["PHANTOM"] = np.concatenate((test_images["PHANTOM"], test_images["VIRTUAL_BREAST"]), axis=0)
		except KeyError:
			train_images["PHANTOM"] = train_images["VIRTUAL_BREAST"]
			test_images["PHANTOM"] = test_images["VIRTUAL_BREAST"]
	if "FDK" in train_images.keys():
		try:
			if merge_datasets:
				train_images["FBP"] = np.concatenate((train_images["FBP"], train_images["FDK"]), axis=0)
				test_images["FBP"] = np.concatenate((test_images["FBP"], test_images["FDK"]), axis=0)
		except KeyError:
			train_images["FBP"] = train_images["FDK"]
			test_images["FBP"] = test_images["FDK"]

	if "DATA_LEO_DIFF" in train_images.keys():
		try:
			if merge_datasets:
				train_images["DIFF"] = np.concatenate((train_images["DIFF"], train_images["DATA_LEO_DIFF"]), axis=0)
				test_images["DIFF"] = np.concatenate((test_images["DIFF"], test_images["DATA_LEO_DIFF"]), axis=0)

		except KeyError:
			train_images["DIFF"] = train_images["DATA_LEO_DIFF"]
			test_images["DIFF"] = test_images["DATA_LEO_DIFF"]

	if "AUG" in train_images.keys() and "AUG_TARGET":
		try:
			train_images["FBP"] = np.concatenate((train_images["FBP"], train_images["AUG"]), axis=0)
			train_images["PHANTOM"] = np.concatenate((train_images["PHANTOM"], train_images["AUG_TARGET"]), axis=0)

		except KeyError:
			train_images["FBP"] = train_images["AUG"]
			train_images["PHANTOM"] = train_images["AUG_TARGET"]
			test_images["FBP"] = test_images["AUG"]
			test_images["PHANTOM"] = test_images["AUG_TARGET"]

	print(train_images.keys())
	# shuffle
	print(len(train_images["FBP"]))
	idx_images_train = list(range(len(train_images["FBP"])))
	np.random.shuffle(idx_images_train)
	idx_images_test = list(range(len(test_images["FBP"])))
	np.random.shuffle(idx_images_test)
	if "DIFF" in train_images.keys():
		train_images["DIFF"] = train_images["DIFF"][idx_images_train]
		test_images["DIFF"] = test_images["DIFF"][idx_images_test]

	train_images["FBP"] = train_images["FBP"][idx_images_train]
	train_images["PHANTOM"] = train_images["PHANTOM"][idx_images_train]

	test_images["FBP"] = test_images["FBP"][idx_images_test]
	test_images["PHANTOM"] = test_images["PHANTOM"][idx_images_test]
	#print(idx_images_train)
	return train_images, test_images
