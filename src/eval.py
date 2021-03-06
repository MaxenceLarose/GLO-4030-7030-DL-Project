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
from model.inceptionNet import InceptionNet
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from data_loader.data_loaders import load_all_images, load_images
from deeplib.training import HistoryCallback, get_model
from model.metrics import validate_model, RMSELoss, PSNRLoss, SSIMLoss
from data_loader.datasets import BreastCTDataset, train_valid_loaders
from draw_images import draw_pred_target

from utils.util import get_preprocessing
from logger.logging_tools import logs_file_setup, log_device_setup, set_seed
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.filters.rank import gradient

def eval_model(
		network,
		test_loader,
		criterion,
		model_weigths_path,
		predicted_images_save_path,
		batch_size=1,
		use_gpu=True,
		save_data=True,
		eval_all_criterion=True
):

	if criterion == "MSELoss":
		loss = nn.MSELoss()
	elif criterion == "RMSELoss":
		loss = RMSELoss()
	elif criterion == "PSNRLoss":
		loss = PSNRLoss()
	elif criterion == "SSIMLoss":
		loss = SSIMLoss()
	else:
		raise RuntimeError("{} criterion not available!".format(criterion))

	model = get_model(network, optimizer=None, criterion=loss, use_gpu=use_gpu)

	model.load_weights(model_weigths_path)

	if use_gpu:
		model.cuda()

	results = validate_model(
		model,
		test_loader,
		save_data=save_data,
		return_images=True,
		output_path=predicted_images_save_path,
		evaluate_worst_RMSE=False
	)
	logging.info(f"{criterion}: {results[0]}.\n")

	predictions, targets = results[1], results[2]

	if eval_all_criterion:

		loss_functions = dict(
			PSNRLoss=psnr,
		)

		for loss_name, loss_function in loss_functions.items():
			losses: list = []
			for target, prediction in zip(targets, predictions):
				if loss_name == "SSIMLoss":
					losses.append(loss_function(target[0, :, :], prediction[0, :, :], win_size=25))
				else:
					losses.append(loss_function(target[0, :, :], prediction[0, :, :]))
			#losses = loss_function(targets, predictions)
			np.savetxt("{}/{}.txt".format(predicted_images_save_path, loss_name), losses)
			mean_loss = np.mean(losses)
			std = np.std(losses)
			logging.info(f"{loss_name} : {mean_loss}")
			logging.info(f"{loss_name} std : {std}\n")


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
	debug = True
	batch_size = 4
	criterion = "RMSELoss"
	optimizer = "Adam"
	eval_train_images = True
	n_data_batch = 4
	save_data = True

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
		"UNetSinogramInterpolator",
		"NestedUNet",
		"InceptionUNet",
		"InceptionNet",
		"SMP UnetPLusPLus",
		"Pretrained Simple UNet",
		"Pretrained RED_CNN",
		"BreastUNet",
		"Pretrained SMP UNet",
		"SMP UNet",
		"BreastCNN",
		"BreastUNet8",
		"BreastUNet2",
		"BreastUNet3",
		"BreastUNet4",
		"BreastUNet5",
		"BreastUNet7"
	]

	# networks_to_use: List[str] = [
	# 	"SMP UNet",
	# 	"UNet",
	# 	"UNet_AUG",
	# 	"NestedUNet",
	# 	"BreastUNet"
	# ]
	networks_to_use: List[str] = [
		"BreastUNet8"
	]
	dataset_to_eval = "test" # or test
	# --------------------------------------------------------------------------------- #
	#                            dataset                                                #
	# --------------------------------------------------------------------------------- #
	preprocessing = None
	if eval_train_images:
		train_images = {}
		if dataset_to_eval == "test":
			train_images_aapm = load_all_images(["TEST_OSC"], n_batch=1)
			train_images["INPUTS"] = train_images_aapm["TEST_OSC"]
			train_images["TARGETS"] = train_images_aapm["TEST_OSC"]
		elif dataset_to_eval == "train":
			train_images_aapm = load_all_images(["OSC_TV_AAPM"], ext=".mha", n_batch=4)
			train_images["INPUTS"] = train_images_aapm["OSC_TV_AAPM"]
			train_images_aapm = load_all_images(["PHANTOM"], n_batch=4)
			train_images["TARGETS"] = train_images_aapm["PHANTOM"]
		else:
			raise RuntimeError("Wrong dataset to eval. The choice is train or test.")
		train_dataset = BreastCTDataset(train_images["INPUTS"], train_images["TARGETS"], preprocessing=preprocessing)
		# if debug:
		# 	aapm_dataset = Subset(aapm_dataset, [0, 1])

		test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

	for network_to_use in networks_to_use:
		if network_to_use not in available_networks:
			raise NotImplementedError(
				f"Chosen network isn't implemented \nImplemented networks are {available_networks}.")
		elif network_to_use == "UNet":
			model = UNet(1,1)
			preprocessing = None
		elif network_to_use == "UNetSinogramInterpolator":
			model = UNet(1,1, filters=[32, 64, 128, 256, 512], sparse_sinogram_net=True)
			preprocessing = None
		elif network_to_use == "Pretrained SMP UNet":
			encoder = "resnet34"
			encoder_weights = "imagenet"
			encoder_depth = 5
			decoder_channels = (512, 256, 128, 64, 32)
			preprocessing = get_preprocessing(get_preprocessing_fn(
				encoder_name=encoder, pretrained=encoder_weights))
			if preprocessing:
				in_channels = 3
			else:
				in_channels = 1
			model = UNetSMP(
				unfreezed_layers=[],
				in_channels=in_channels,
				encoder=encoder,
				encoder_depth=encoder_depth,
				decoder_channels=decoder_channels,
				encoder_weights=encoder_weights,
				activation=None
			)
		elif network_to_use == "SMP UNet":
			encoder = "resnet34"
			encoder_weights = None
			encoder_depth = 5
			decoder_channels = (512, 256, 128, 64, 32)
			preprocessing = None
			if preprocessing:
				in_channels = 1
			else:
				in_channels = 1
			model = UNetSMP(
				unfreezed_layers=[],
				in_channels=in_channels,
				encoder=encoder,
				encoder_depth=encoder_depth,
				decoder_channels=decoder_channels,
				encoder_weights=encoder_weights,
				activation=None
			)
		elif network_to_use == "NestedUNet":
			nb_filter=(32, 64, 128, 256, 512)
			model = NestedUNet(1,1, nb_filter=nb_filter, batch_norm_momentum=batch_norm_momentum)
			preprocessing = None
		elif network_to_use == "InceptionUNet":
			nb_filter=(32, 64, 128, 256, 512)
			model = InceptionUNet(1,1, nb_filter=nb_filter, batch_norm_momentum=batch_norm_momentum, kernel_size_1=[5,5,5,5,5], kernel_size_2=[3,3,3,3,3])
			preprocessing = None
		elif network_to_use == "InceptionNet":
			model = InceptionNet(1, 1, 64, n_inception_blocks=5, batch_norm_momentum=batch_norm_momentum,
				use_maxpool=True)
			preprocessing = None
		elif "BreastUNet" in network_to_use:
			model = BreastCNN(1, 1, norm_momentum=batch_norm_momentum, middle_channels=[32, 64, 128], unet_arch=True, upsample_mode="bicubic")
			preprocessing = None
		elif network_to_use == "BreastCNN":
			model = BreastCNN(1, 1, batch_norm_momentum=batch_norm_momentum, middle_channels=[32, 64, 128], unet_arch=False)
			preprocessing = None
		elif network_to_use == "SMP UnetPLusPLus":
			encoder = "resnet34"
			encoder_weights = "imagenet"
			activation = "logits"
			encoder_depth = 5
			decoder_channels = (512, 256, 128, 64, 32)
			preprocessing = get_preprocessing(get_preprocessing_fn(
				encoder_name=encoder, pretrained=encoder_weights))
			if preprocessing:
				in_channels = 3
			else:
				in_channels = 1
			model = UNetPlusPLus(
				unfreezed_layers=["encoder", "decoder", "segmentation_head"],
				in_channels=in_channels,
				encoder=encoder,
				encoder_depth=encoder_depth,
				decoder_channels=decoder_channels,
				encoder_weights=encoder_weights,
				activation=None
			)
		elif network_to_use == "Pretrained Simple UNet":
			model = PretrainedUNet(
				1, unfreezed_layers=["inc", "down1", "down2", "down3", "down4", "down5", "up1", "up2", "up3", "up4", "up5", "outc"]
			)
			preprocessing = None
		elif network_to_use == "Pretrained RED_CNN":
			model = PretrainedREDCNN(unfreezed_layers=["conv", "tconv"])
			preprocessing = None
		else:
			warnings.warn("Something very wrong happened")
		logging.info(f"\nNombre de param??tres: {np.sum([p.numel() for p in model.parameters()])}")
		# --------------------------------------------------------------------------------- #
		#                            network prediction                                     #
		# --------------------------------------------------------------------------------- #
		model_weigths_path = "model/models_weights_challenge/{}_weights_best.pt".format(network_to_use)
		if dataset_to_eval == "test":
			save_path_for_predictions = "results_challenge/yolo/{}/train_images_prediction".format(network_to_use)
		elif dataset_to_eval == "train":
			save_path_for_predictions = "results_challenge/{}/train_images_prediction".format(network_to_use)
		else:
			raise RuntimeError("Wrong dataset to eval. The choice is train or test.")

		logging.info(f"Begin validation of network {network_to_use}.")
		eval_model(
			network=model,
			test_loader=test_loader,
			criterion=criterion,
			model_weigths_path=model_weigths_path,
			save_data=save_data,
			predicted_images_save_path=save_path_for_predictions,
			use_gpu=use_gpu
		)

		if debug:
			files_target = os.path.join(save_path_for_predictions, "{}/{}.npy".format("ref", "targets"))
			files_predicts = os.path.join(save_path_for_predictions, "{}/{}.npy".format("res", "predictions"))

			targets_images = np.load(files_target)
			predictions_images = np.load(files_predicts)

			random_idx = 42
			target_image = targets_images[random_idx]
			pred_image = predictions_images[random_idx]

			fig, ax = plt.subplots()
			size_split = 10
			# pred - target
			diff = pred_image 
			im1 = ax.imshow(target_image, cmap='Greys')
			logging.info(f"The mean diff is {np.mean(diff)}.")
			divider1 = make_axes_locatable(ax)
			cax1 = divider1.append_axes("right", size="{}%".format(size_split), pad=0.05)
			cbar1 = plt.colorbar(im1, cax=cax1)
			ax.set_xticks([])
			ax.set_yticks([])
			#ax.set_title("Target")
			plt.savefig(os.path.join(save_path_for_predictions, "diff_{}.jpg".format(random_idx)))
			#plt.show()
			fig, ax = plt.subplots()
			# prediction
			im2 = ax.imshow(pred_image, cmap='Greys')
			divider2 = make_axes_locatable(ax)
			cax2 = divider2.append_axes("right", size="{}%".format(size_split), pad=0.05)
			cbar2 = plt.colorbar(im2, cax=cax2)
			ax.set_title("Prediction")
			plt.show()
