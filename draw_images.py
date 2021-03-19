# Standard lib python import
import os

# Specialized python lib
import numpy as np
import matplotlib.pyplot as plt

# Local project import
from datasets import load_all_images, BreastCTDataset, train_valid_loaders

# pytorch
from torch.utils.data import Dataset



def draw_images(images : dict, image_idx=0):
	fig, ax = plt.subplots(1, 3)
	i = 0
	for image_type, image_data in images.items():
		ax[i].imshow(image_data[image_idx])
		ax[i].set_title(image_type)
		i += 1
	plt.show()

def draw_data_targets(dataset : Dataset, image_idx=0):
	fig, ax = plt.subplots(1, 2)
	data, targets = dataset.to_numpy()
	ax[0].imshow(data[image_idx])
	ax[0].set_title("Data")
	ax[1].imshow(targets[image_idx])
	ax[1].set_title("Target")
	plt.show()

if __name__ == '__main__':
	train_images, test_images = load_all_images(n_batch=1)
	breast_CT_dataset_train = BreastCTDataset(train_images["FBP"], train_images["PHANTOM"])
	draw_data_targets(breast_CT_dataset_train)
	draw_images(train_images)