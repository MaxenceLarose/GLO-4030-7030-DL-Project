import logging
import time
import os
import warnings
import numpy as np

import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from model.unet import UNet
from model.inceptionUnet import InceptionUNet
from model.nestedUnet import NestedUNet
from model.pretrained_unet import PretrainedUNet
from model.segmentation_models import UNetSMP, UNetPlusPLus
from model.RED_CNN import PretrainedREDCNN
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from deeplib.history import History
from deeplib.training import HistoryCallback, get_model
from poutyne import ModelCheckpoint, ReduceLROnPlateau

from utils.util import get_preprocessing
from draw_images import draw_pred_target, draw_all_preds_targets
from model.metrics import validate_model, RMSELoss
from data_loader.data_loaders import load_all_images, load_images
from data_loader.datasets import BreastCTDataset, train_valid_loaders
from logger.logging_tools import logs_file_setup, log_device_setup, set_seed


# Fonction qui provient de la librairie deeplib (https://github.com/ulaval-damas/glo4030-labs/tree/master/deeplib)
def train_network(
		network,
		dataset,
		*,
		optimizer="Adam",
		lr=0.001,
		momentum=0.9,
		weight_decay=0,
		n_epoch=5,
		batch_size=1,
		use_gpu=True,
		criterion="MSELoss",
		callbacks=None,
		load_network_state=False,
		save_path="model/model_weights",
		dataset_test_challenge=None,
		leonardo_dataset=None
):
	"""
	Entraîne un réseau de neurones PyTorch avec Poutyne. On suppose que la sortie du réseau est
	compatible avec la fonction cross-entropy de PyTorch pour calculer l'exactitude (accuracy).

	Args:
		network (nn.Module): Un réseau de neurones PyTorch
		dataset (torch.utils.data.Dataset): Un jeu de données PyTorch
		optimizer (string): Le nom d'un optimiseur PyTorch
		lr (float): Le learning rate pour l'entraînement
		momentum (float): Le momentum pour l'entraînement (si SGD)
		weight_decay (float): Le weight decay pour l'entraînement
		n_epoch (int): Le nombre d'epochs d'entraînement désiré
		batch_size (int): La taille de batch désirée
		use_gpu (bool): S'il faut utiliser le GPU (avertissement si pas de GPU)
		criterion (string): Le nom d'une fonction de perte compatible avec la cross-entropy de PyTorch
		callbacks (List[poutyne.Callback]): Une liste de callbacks de Poutyne
		load_network_state (bool): S'il faut charger un fichier de poids plutôt qu'entraîner
		save_path (string): Le répertoire où sauvegarder les poids du réseau
		dataset_test_challenge (torch.utils.data.Dataset): Le jeu de données de test du concours
		leonardo_dataset (torch.utils.data.Dataset): Le jeu de données pour l'augmentation

	Returns:
		Retourne un objet de type `deeplib.history.History` contenant l'historique d'entraînement.
	"""
	# --------------------------------------------------------------------------------- #
	#                                 build model                                       #
	# --------------------------------------------------------------------------------- #
	history_callback = HistoryCallback()
	checkpoint_callback = ModelCheckpoint("{}_best.pt".format(save_path), save_best_only=True)
	scheduler = ReduceLROnPlateau(patience=4, factor=0.1)
	callbacks = [
		history_callback, checkpoint_callback, scheduler] if callbacks is None else [
		history_callback, checkpoint_callback, scheduler] + callbacks
	if leonardo_dataset is not None:
		train_loader, valid_loader = train_valid_loaders(
			dataset, batch_size=batch_size, valid_dataset=leonardo_dataset)
	else:
		train_loader, valid_loader = train_valid_loaders(
			dataset, batch_size=batch_size, train_split=0.9)
	if optimizer == "Adam":
		opt = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
	elif optimizer == "SGD":
		opt = optim.SGD(network.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
	else:
		raise RuntimeError("{} optimizer not available!".format(optimizer))
	if criterion == "MSELoss":
		loss = nn.MSELoss()
	elif criterion == "RMSELoss":
		loss = RMSELoss()
	else:
		raise RuntimeError("{} criterion not available!".format(optimizer))
	model = get_model(network, optimizer=opt, criterion=loss, use_gpu=use_gpu)
	if not load_network_state:
		# --------------------------------------------------------------------------------- #
		#                                 train model                                       #
		# --------------------------------------------------------------------------------- #
		model.fit_generator(
			train_loader,
			valid_loader,
			epochs=n_epoch,
			progress_options=dict(coloring=False),
			callbacks=callbacks)
		# --------------------------------------------------------------------------------- #
		#                            save model at the end                                  #
		# --------------------------------------------------------------------------------- #
		model.save_weights("{}_end.pt".format(save_path))
	# --------------------------------------------------------------------------------- #
	#                               load best model                                     #
	# --------------------------------------------------------------------------------- #
	model.load_weights("{}_best.pt".format(save_path))
	# --------------------------------------------------------------------------------- #
	#                            save challenge images                                  #
	# --------------------------------------------------------------------------------- #
	if dataset_test_challenge is not None:
		test_loader = DataLoader(dataset_test_challenge, batch_size=batch_size)
		validate_model(model, test_loader, save_data=True, output_path="data/challenge")
		#draw_all_preds_targets(network, test_loader, os.path.relpath("../Figure_challenge"))
	# --------------------------------------------------------------------------------- #
	#                            save validation images                                 #
	# --------------------------------------------------------------------------------- #
	validate_model(model, valid_loader, save_data=True)
	#draw_all_preds_targets(network, valid_loader)
	return history_callback.history


if __name__ == '__main__':
	# --------------------------------------------------------------------------------- #
	#                            Logs Setup                                             #
	# --------------------------------------------------------------------------------- #
	logs_file_setup(__file__, logging.INFO)
	log_device_setup()

	# --------------------------------------------------------------------------------- #
	#                            Constants                                              #
	# --------------------------------------------------------------------------------- #
	# training setup constants
	load_data_for_challenge = True
	load_network_state = False
	lr = 0.0001
	momentum = 0.9
	n_epoch = 150
	batch_size = 1
	weight_decay = 1e-4
	criterion = "RMSELoss"
	optimizer = "Adam"
	# unet setup constants
	if batch_size == 1:
		batch_norm_momentum = 0.01
	else:
		batch_norm_momentum = 0.05
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
		"Pretrained RED_CNN"
	]
	network_to_use: str = "UNet"
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
		model = InceptionUNet(1,1, nb_filter=nb_filter, batch_norm_momentum=batch_norm_momentum, kernel_size_1=[7,5,3,3,3], kernel_size_2=[5,3,3,3,1])
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
			activation=activation
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
	if load_data_for_challenge:
		train_images, test_images = load_all_images(
			image_types=["FBP", "Phantom", "Sinogram"], n_batch=4,
			multiple_channels=False, load_sinograms=False, merge_datasets=False)
		valid_images_contest = load_images("FBP", path="data/validation", n_batch=1)
		aapm_dataset = BreastCTDataset(
			train_images["FBP"], train_images["PHANTOM"], preprocessing=preprocessing)
		aapm_dataset_valid = BreastCTDataset(
			valid_images_contest, valid_images_contest, preprocessing=preprocessing)
		leonardo_dataset = None
		# leonardo_dataset = BreastCTDataset(
		# 	train_images["FDK"][:100], train_images["VIRTUAL_BREAST"][:100], preprocessing=preprocessing)
	else:
		train_images, test_images = load_all_images(
			image_types=["FBP", "Phantom", "Sinogram", "virtual_breast", "fdk"], n_batch=1)
		aapm_dataset = BreastCTDataset(
			train_images["FBP"], train_images["PHANTOM"], preprocessing=preprocessing)
		leonardo_dataset = BreastCTDataset(
			train_images["FDK"], train_images["VIRTUAL_BREAST"], preprocessing=preprocessing)
		aapm_dataset_valid = None

	# --------------------------------------------------------------------------------- #
	#                           network training                                        #
	# --------------------------------------------------------------------------------- #
	history = train_network(
		model,
		aapm_dataset,
		optimizer=optimizer,
		lr=lr,
		momentum=momentum,
		weight_decay=weight_decay,
		n_epoch=n_epoch,
		batch_size=batch_size,
		criterion=criterion,
		use_gpu=True,
		load_network_state=load_network_state,
		save_path="model/{}_weights".format(network_to_use),
		dataset_test_challenge=aapm_dataset_valid,
		leonardo_dataset=leonardo_dataset
	)

	# --------------------------------------------------------------------------------- #
	#                           network analysing                                       #
	# --------------------------------------------------------------------------------- #
	if not load_network_state:
		history.display()
