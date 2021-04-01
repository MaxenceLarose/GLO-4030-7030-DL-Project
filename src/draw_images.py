# Standard lib python import
import os

# Specialized python lib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Local project import
from data_loader.data_loaders import load_all_images
from data_loader.datasets import BreastCTDataset

# pytorch
from torch.utils.data import Dataset

from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity


def draw_images(images : dict, image_idx=0):
	fig, ax = plt.subplots(1, 3)
	i = 0
	for image_type, image_data in images.items():
		ax[i].imshow(image_data[image_idx][0])
		ax[i].set_title(image_type)
		i += 1
	plt.show()


def draw_data_targets(dataset : Dataset, image_idx=0):
	data, targets = dataset.to_numpy()
	fig, ax = plt.subplots(1, 2)
	ax[0].imshow(data[image_idx][0])
	ax[0].set_title("Data")
	ax[1].imshow(targets[image_idx][0])
	ax[1].set_title("Target")
	plt.show()


def draw_pred_target(inputs, targets, pred, image_idx=0, fig_id=0, output_path=os.path.relpath("../Figures")):
	fig, ax = plt.subplots(2, 3, figsize=(20,10))
	size_split = 10
	input_image = inputs[image_idx][0]
	target_image = targets[image_idx][0]
	pred_image = pred[image_idx][0]

	# input
	im1 = ax[0,0].imshow(input_image, cmap='Greys')
	divider1 = make_axes_locatable(ax[0,0])
	cax1 = divider1.append_axes("right", size="{}%".format(size_split), pad=0.05)
	cbar1 = plt.colorbar(im1, cax=cax1)
	ax[0,0].set_title("Input")

	# target
	im2 = ax[0,1].imshow(target_image	, cmap='Greys')
	divider2 = make_axes_locatable(ax[0,1])
	cax2 = divider2.append_axes("right", size="{}%".format(size_split), pad=0.05)
	cbar2 = plt.colorbar(im2, cax=cax2)
	ax[0,1].set_title("Target")

	# prediction
	im3 = ax[1,0].imshow(pred_image, cmap='Greys')
	divider3 = make_axes_locatable(ax[1,0])
	cax3 = divider3.append_axes("right", size="{}%".format(size_split), pad=0.05)
	cbar3 = plt.colorbar(im3, cax=cax3)
	ax[1,0].set_title("Prediction")

	diff = pred_image - target_image
	rmse = np.sqrt(np.mean((diff**2)))

	# SSIM is defined for positive values
	min_pred = np.min(pred_image)
	pred_image -= min_pred	
	target_image -= min_pred
	mssim, ssim = structural_similarity(pred_image, target_image, full=True)
	im4 = ax[1,1].imshow(ssim, cmap='Greys')
	divider4 = make_axes_locatable(ax[1,1])
	cax4 = divider4.append_axes("right", size="{}%".format(size_split), pad=0.05)
	cbar4 = plt.colorbar(im4, cax=cax4)
	ax[1,1].set_title("SSIM")
	
	# Diff
	im5 = ax[0,2].imshow(diff, cmap='Greys')
	divider5 = make_axes_locatable(ax[0,2])
	cax5 = divider5.append_axes("right", size="{}%".format(size_split), pad=0.05)
	cbar5 = plt.colorbar(im5, cax=cax5)
	ax[0,2].set_title("Difference")

	# Histogram of diff
	ax[1,2].hist(diff, bins=100)
	ax[1,2].set_xlabel("Difference")

	#plt.show()
	if not os.path.isdir(output_path):
		os.makedirs(output_path)
	plt.savefig("{}/pred_valid_{}".format(output_path, fig_id), bbox_inches='tight')
	plt.close(fig)
	return mssim, rmse

def draw_pixel_value_histogram(data : np.ndarray):
	tmp_data = copy.deepcopy(data)
	tmp_data = tmp_data.reshape(tmp_data.shape[0], tmp_data.shape[2], tmp_data.shape[3])
	min_val = np.min(tmp_data)
	max_val = np.max(tmp_data)
	print(min_val, max_val)
	hist_values = []
	for i in range(tmp_data.shape[0]):
		values, bin_edges = np.histogram(tmp_data[i], bins=100, range=(min_val, max_val))
		hist_values.append(values)
	hist_values = np.array(hist_values)
	hist_values = np.sum(hist_values, axis=0)
	fig, ax = plt.subplots()
	ax.plot(bin_edges[:-1], hist_values, "bo")
	ax.set_xlim(min_val, max_val)
	ax.set_yscale('log')
	plt.show()


if __name__ == '__main__':
	train_images, test_images = load_all_images(n_batch=1)
	draw_pixel_value_histogram(train_images["FBP"])
	breast_CT_dataset_train = BreastCTDataset(train_images["FBP"], train_images["PHANTOM"])
	draw_data_targets(breast_CT_dataset_train)
	draw_images(train_images)
