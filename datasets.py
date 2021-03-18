# Standard lib python import
import glob
import os

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
	files = glob.glob(os.path.join(path, "{}*.npy.gz".format(image_type)))
	data = []
	for file in files[:n_batch]:
		f = gzip.GzipFile(file, "r")
		batch = np.load(f)
		data.append(batch)
	data = np.concatenate(tuple(data))
	return data

def load_all_images(path : str="data", n_batch=4) -> dict:
	"""
	Read all the images located in the folder path. 
	The dtype are already encoded in float32.

	args:
		path (str): the path to the data files

	return:	
		The image dataset (dict)
	"""
	images = {}
	image_types = ["Sinogram", "FBP", "Phantom"]
	for image_type in image_types:
		images[image_type.upper()] = load_images(image_type, n_batch=n_batch)
	return images
