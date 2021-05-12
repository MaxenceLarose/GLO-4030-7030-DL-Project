import logging
import time
import os
import warnings
import numpy as np
import pprint

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
from model.BREAST_CNN import BreastCNN
from model.inceptionNet import InceptionNet
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from utils.data_augmentation import save_augmented_dataset
from model.FCSinogramReconstruction import LinearWS, FCSinogramReconstruction


from deeplib.history import History
from deeplib.training import HistoryCallback, get_model
from poutyne import ModelCheckpoint, ReduceLROnPlateau, ExponentialLR

from utils.util import get_preprocessing
from draw_images import draw_pred_target, draw_all_preds_targets, draw_data_targets_2	
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
		lr_decay=0.94,
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
		valid_dataset=None
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
		valid_dataset (torch.utils.data.Dataset): Le jeu de données pour l'augmentation

	Returns:
		Retourne un objet de type `deeplib.history.History` contenant l'historique d'entraînement.
	"""
	# --------------------------------------------------------------------------------- #
	#                                 build model                                       #
	# --------------------------------------------------------------------------------- #
	history_callback = HistoryCallback()
	checkpoint_callback = ModelCheckpoint("{}_best.pt".format(save_path), save_best_only=True)
	# if lr_decay	<= 0:
	# 	scheduler = ReduceLROnPlateau(patience=3, factor=0.333)
	# else:
	# 	scheduler = ExponentialLR(lr_decay)
	scheduler = ExponentialLR(lr_decay)
	callbacks = [
		history_callback, checkpoint_callback, scheduler] if callbacks is None else [
		history_callback, checkpoint_callback, scheduler] + callbacks
	if valid_dataset is not None:
		train_loader, _ = train_valid_loaders(
			dataset, batch_size=batch_size, train_split=1)
		valid_loader, _ = train_valid_loaders(
			valid_dataset, batch_size=batch_size, train_split=1)
	else:
		train_loader, valid_loader = train_valid_loaders(
			dataset, batch_size=batch_size, train_split=0.95)
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
	log_device_setup()
	#model.load_weights("{}_best.pt".format(save_path))
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
	with torch.no_grad():
		for i, (inputs, targets) in enumerate(valid_loader):
			if use_gpu:
				inputs = inputs.cuda()
				targets = targets.cuda()
			res = model.evaluate_on_batch(inputs, targets, return_pred=True)
			targets = targets.cpu().numpy()
			draw_data_targets_2(res[-1], targets, image_idx=0)
			exit(0)
	#if dataset_test_challenge is not None:
	#	test_loader = DataLoader(dataset_test_challenge, batch_size=batch_size)
		#validate_model(model, test_loader, save_data=True, output_path="data/challenge", evaluate_worst_RMSE=False)
		#draw_all_preds_targets(network, test_loader, os.path.relpath("../Figure_challenge"))
	# --------------------------------------------------------------------------------- #
	#                            save validation images                                 #
	# --------------------------------------------------------------------------------- #
	#validate_model(model, valid_loader, save_data=True, evaluate_worst_RMSE=False)
	#draw_all_preds_targets(network, valid_loader)
	return history_callback.history


if __name__ == '__main__':
	# --------------------------------------------------------------------------------- #
	#                            Logs Setup                                             #
	# --------------------------------------------------------------------------------- #
	logs_file_setup(__file__, logging.INFO)
	# log_device_setup()
	#save_augmented_dataset()
	#exit(0)
	# --------------------------------------------------------------------------------- #
	#                            Constants                                              #
	# --------------------------------------------------------------------------------- #
	# training setup constants
	use_gpu = True
	load_data_for_challenge = True
	load_network_state = False
	lr = 0.0001
	momentum = 0.9
	n_epoch = 300
	batch_size = 2
	weight_decay = 1e-4
	criterion = "RMSELoss"
	optimizer = "Adam"
	lr_decay = 0.99
	# unet setup constants
	if batch_size == 1:
		batch_norm_momentum = 0.01
	elif batch_size <= 8:
		batch_norm_momentum = 0.05
	else:
		batch_norm_momentum = 0.1
	norm = "BN"
	num_groups = 8
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
		"InceptionNet",
		"SMP UnetPLusPLus",
		"Pretrained Simple UNet",
		"Pretrained RED_CNN",
		"BreastUNet",
		"Pretrained SMP UNet",
		"SMP UNet",
		"BreastCNN",
		"LinearWS"
	]
	network_to_use: str = "BreastUNet"
	if network_to_use not in available_networks:
		raise NotImplementedError(
			f"Chosen network isn't implemented \nImplemented networks are {available_networks}.")
	elif network_to_use == "UNet":
		model = UNet(1,1, filters=[32, 64, 128, 256, 512], norm=norm, num_groups=num_groups)
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
			unfreezed_layers=["decoder", "segmentation_head"],
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
			unfreezed_layers=["encoder", "decoder", "segmentation_head"],
			in_channels=in_channels,
			encoder=encoder,
			encoder_depth=encoder_depth,
			decoder_channels=decoder_channels,
			encoder_weights=encoder_weights,
			activation=None
		)
	elif network_to_use == "NestedUNet":
		nb_filter=(64, 128, 256, 512, 1024)
		model = NestedUNet(1,1, nb_filter=nb_filter, batch_norm_momentum=batch_norm_momentum, deep_survervision=True)
		preprocessing = None
	elif network_to_use == "InceptionUNet":
		nb_filter=(32, 64, 128, 256, 512)
		model = InceptionUNet(1,1, nb_filter=nb_filter, batch_norm_momentum=batch_norm_momentum, kernel_size_1=[5,5,5,5,5], kernel_size_2=[3,3,3,3,3])
		preprocessing = None
	elif network_to_use == "InceptionNet":
		model = InceptionNet(1, 1, 64, n_inception_blocks=5, batch_norm_momentum=batch_norm_momentum,
			use_maxpool=True)
		preprocessing = None
	elif network_to_use == "BreastUNet":
		model = BreastCNN(1, 1, norm_momentum=batch_norm_momentum, middle_channels=[64, 128, 256], unet_arch=True)
		preprocessing = None
	elif network_to_use == "BreastCNN":
		model = BreastCNN(1, 1, norm_momentum=batch_norm_momentum, middle_channels=[16, 32, 64], unet_arch=False)
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
	elif network_to_use == "LinearWS":
		model = LinearWS(128, 1024, 512)
		#model = FCSinogramReconstruction(128, 1024, 512, 5, 256)
		preprocessing = None
	else:
		warnings.warn("Something very wrong happened")
	logging.info(f"\nNombre de paramètres: {np.sum([p.numel() for p in model.parameters()])}")
	#exit(0)
	# --------------------------------------------------------------------------------- #
	#                            dataset                                                #
	# --------------------------------------------------------------------------------- #
	# train
	train_images = {}
	train_images_aapm = load_all_images(["FBP128", "PHANTOM"], n_batch=4, flatten_images=False)
	#train_images_leo = load_all_images(["AUG_BATCH", "AUG_TARGET"], n_batch=3, flatten_images=False)
	train_images["INPUTS"] = train_images_aapm["FBP128"][:3950]
	train_images["TARGETS"] = train_images_aapm["PHANTOM"][:3950]
	# train_images["INPUTS"] = np.concatenate((train_images_aapm["FBP128"][:3950], train_images_leo["AUG_BATCH"]))
	# train_images["TARGETS"] = np.concatenate((train_images_aapm["PHANTOM"][:3600], train_images_leo["AUG_TARGET"]))
	train_dataset = BreastCTDataset(train_images["INPUTS"], train_images["TARGETS"], preprocessing=preprocessing)
	#print(train_images_aapm["SINOGRAM"].shape)
	# train_dataset = BreastCTDataset(train_images_aapm["FBP"][:100], train_images_aapm["PHANTOM"][:100], preprocessing=preprocessing)
	# print(len(train_dataset))

	# valid
	valid_images = {}
	valid_images["INPUTS"] = train_images_aapm["FBP128"][3950:]
	valid_images["TARGETS"] = train_images_aapm["PHANTOM"][3950:]

	#valid_images["INPUTS"] = np.concatenate((train_images_aapm["FBP128"][1500:], train_images_leo["FBP_"][1500:]))
	#valid_images["TARGETS"] = np.concatenate((train_images_aapm["PHANTOM"][500:], train_images_leo["VIRTUAL_BREAST"][3500:]))
	valid_dataset = BreastCTDataset(valid_images["INPUTS"], valid_images["TARGETS"], preprocessing=preprocessing)

	#valid_dataset = None
	valid_images_contest = None

	# --------------------------------------------------------------------------------- #
	#                           network training                                        #
	# --------------------------------------------------------------------------------- #
	history = train_network(
		model,
		train_dataset,
		optimizer=optimizer,
		lr=lr,
		lr_decay=lr_decay,
		momentum=momentum,
		weight_decay=weight_decay,
		n_epoch=n_epoch,
		batch_size=batch_size,
		criterion=criterion,
		use_gpu=use_gpu,
		load_network_state=load_network_state,
		save_path="model/models_weights/{}_weights".format(network_to_use),
		dataset_test_challenge=valid_images_contest,
		valid_dataset=valid_dataset
	)

	# --------------------------------------------------------------------------------- #
	#                           network analysing                                       #
	# --------------------------------------------------------------------------------- #
	if not load_network_state:
		logging.info(f"history: \n{pprint.pformat(history.history, indent=4)}")
		history.display()
