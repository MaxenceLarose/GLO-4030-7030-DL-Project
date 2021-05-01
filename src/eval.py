import os
import logging
import warnings
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List

import torch.nn as nn
import torch
from torch.utils.data import Subset, Dataset, DataLoader
import torch.optim as optim

from model.unet import UNet
from model.inceptionUnet import InceptionUNet
from model.nestedUnet import NestedUNet
from model.pretrained_unet import PretrainedUNet
from model.segmentation_models import UNetSMP, UNetPlusPLus
from model.RED_CNN import PretrainedREDCNN
from model.BREAST_CNN import BreastCNN
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from data_loader.data_loaders import load_all_images, load_images
from deeplib.training import HistoryCallback, get_model
from model.metrics import validate_model, RMSELoss
from data_loader.datasets import BreastCTDataset, train_valid_loaders
from draw_images import draw_pred_target

from utils.util import get_preprocessing
from logger.logging_tools import logs_file_setup, log_device_setup, set_seed


def eval_model(
		network,
		test_loader,
		criterion,
		model_weigths_path,
		predicted_images_save_path,
		batch_size=1,
		use_gpu=True
):

	if criterion == "MSELoss":
		loss = nn.MSELoss()
	elif criterion == "RMSELoss":
		loss = RMSELoss()
	else:
		raise RuntimeError("{} criterion not available!".format(criterion))

	model = get_model(network, optimizer=None, criterion=loss, use_gpu=use_gpu)

	model.load_weights(model_weigths_path)

	if use_gpu:
		model.cuda()

	validate_model(
		model,
		test_loader,
		save_data=True,
		output_path=predicted_images_save_path,
		evaluate_worst_RMSE=False
	)


if __name__ == '__main__':
	# --------------------------------------------------------------------------------- #
	#                            Logs Setup                                             #
	# --------------------------------------------------------------------------------- #
	logs_file_setup(__file__, logging.INFO)
	log_device_setup()

	# --------------------------------------------------------------------------------- #
	#                            Constants                                              #
	# --------------------------------------------------------------------------------- #
	use_gpu = True
	debug = False
	batch_size = 1
	criterion = "RMSELoss"
	optimizer = "Adam"
	eval_train_images = True
	n_data_batch = 4

	# unet setup constants
	if batch_size == 1:
		batch_norm_momentum = 0.01
	elif batch_size <= 8:
		batch_norm_momentum = 0.05
	else:
		batch_norm_momentum = 0.1

	# seed
	seed = 42
	set_seed(seed)

	# --------------------------------------------------------------------------------- #
	#                            network                                                #
	# --------------------------------------------------------------------------------- #
	available_networks = [
		"UNet",
		"NestedUNet",
		"InceptionUNet",
		"SMP UnetPLusPLus",
		"Pretrained Simple UNet",
		"Pretrained RED_CNN",
		"BreastCNN"
	]

	networks_to_use: List[str] = [
		"UNet",
		"NestedUNet",
		"InceptionUNet",
		"Pretrained RED_CNN"
	]

	for network_to_use in networks_to_use:
		if network_to_use not in available_networks:
			raise NotImplementedError(
				f"Chosen network isn't implemented \nImplemented networks are {available_networks}.")
		elif network_to_use == "UNet":
			nb_filter=(64, 128, 256, 512, 1024)
			model = UNet(1, 1,
						channels_depth_number=nb_filter,
						use_relu=False,  # If False, then LeakyReLU
						mode='nearest',  # For upsampling
						residual_block=True,  # skip connections?
						batch_norm_momentum=batch_norm_momentum)
			preprocessing = None
		elif network_to_use == "NestedUNet":
			nb_filter=(64, 128, 256, 512, 1024)
			model = NestedUNet(1,1, nb_filter=nb_filter, batch_norm_momentum=batch_norm_momentum)
			preprocessing = None
		elif network_to_use == "InceptionUNet":
			nb_filter=(64, 128, 256, 512, 1024)
			model = InceptionUNet(1, 1, nb_filter=nb_filter, batch_norm_momentum=batch_norm_momentum, kernel_size_1=[5,5,3,3,1], kernel_size_2=[3,3,3,3,1])
			preprocessing = None
		elif network_to_use == "BreastCNN":
			model = BreastCNN(1, 1, batch_norm_momentum=batch_norm_momentum, middle_channels=[32, 64, 128])
			preprocessing = None
		elif network_to_use == "SMP UnetPLusPLus":
			encoder = "densenet121"
			encoder_weights = "imagenet"
			activation = "logits"
			encoder_depth = 5
			decoder_channels = (1024, 512, 256, 128, 64)
			preprocessing = get_preprocessing(get_preprocessing_fn(
				encoder_name=encoder, pretrained=encoder_weights))
			if preprocessing:
				in_channels = 3
			else:
				in_channels = 1
			model = UNetPlusPLus(
				unfreezed_layers=["encoder", "decoder"],
				in_channels=in_channels,
				encoder=encoder,
				encoder_depth=encoder_depth,
				decoder_channels=decoder_channels,
				encoder_weights=encoder_weights,
				activation=None
			)
		elif network_to_use == "Pretrained Simple UNet":
			model = PretrainedUNet(
				1, unfreezed_layers=["up1", "up2", "up3", "up4", "up5", "outc"]
			)
			preprocessing = None
		elif network_to_use == "Pretrained RED_CNN":
			model = PretrainedREDCNN(unfreezed_layers=["conv", "tconv"])
			preprocessing = None
		else:
			warnings.warn("Something very wrong happened")
		logging.info(f"\nNombre de paramètres: {np.sum([p.numel() for p in model.parameters()])}")

		# --------------------------------------------------------------------------------- #
		#                            dataset                                                #
		# --------------------------------------------------------------------------------- #
		if eval_train_images:
			train_images, _ = load_all_images(n_batch=n_data_batch)
			aapm_dataset = BreastCTDataset(train_images["FBP"], train_images["PHANTOM"], preprocessing=preprocessing)

			if debug:
				aapm_dataset = Subset(aapm_dataset, [0, 1])

			test_loader = DataLoader(aapm_dataset, batch_size=batch_size, shuffle=False)

		# --------------------------------------------------------------------------------- #
		#                            network prediction                                     #
		# --------------------------------------------------------------------------------- #
		model_weigths_path = "model/models_weights/{}_weights_best.pt".format(network_to_use)
		save_path_for_predictions = "results/{}/train_images_prediction".format(network_to_use)

		logging.info(f"Begin validation of network {network_to_use}.")
		eval_model(
			network=model,
			test_loader=test_loader,
			criterion=criterion,
			model_weigths_path=model_weigths_path,
			predicted_images_save_path=save_path_for_predictions,
			use_gpu=use_gpu
		)

		if debug:
			files_target = os.path.join(save_path_for_predictions, "{}/{}.npy".format("ref", "targets"))
			files_predicts = os.path.join(save_path_for_predictions, "{}/{}.npy".format("res", "predictions"))

			targets_images = np.load(files_target)
			predictions_images = np.load(files_predicts)

			random_idx = np.random.randint(len(targets_images))
			target_image = targets_images[random_idx]
			pred_image = predictions_images[random_idx]

			fig, ax = plt.subplots(1, 2, figsize=(12, 8))
			size_split = 10
			# target
			im1 = ax[0].imshow(target_image, cmap='Greys')
			divider1 = make_axes_locatable(ax[0])
			cax1 = divider1.append_axes("right", size="{}%".format(size_split), pad=0.05)
			cbar1 = plt.colorbar(im1, cax=cax1)
			ax[0].set_title("Target")

			# prediction
			im2 = ax[1].imshow(pred_image, cmap='Greys')
			divider2 = make_axes_locatable(ax[1])
			cax2 = divider2.append_axes("right", size="{}%".format(size_split), pad=0.05)
			cbar2 = plt.colorbar(im2, cax=cax2)
			ax[1].set_title("Prediction")
			plt.show()
