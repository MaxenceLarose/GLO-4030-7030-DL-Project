# Standard lib python import
import os

# Specialized python lib
import numpy as np
import matplotlib.pyplot as plt

# Local project import
from datasets import load_all_images


def draw_image(images : dict, image_idx=0):
	fig, ax = plt.subplots(1, 3)
	i = 0
	for image_type, image_data in images.items():
		ax[i].imshow(image_data[image_idx])
		ax[i].set_title(image_type)
		i += 1
	plt.show()

if __name__ == '__main__':
	images = load_all_images(n_batch=1)
	draw_image(images)