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
from model.nestedUnet import NestedUNet
from model.pretrained_unet import PretrainedUNet
from model.segmentation_models import UNetSMP, UNetPlusPLus
from model.RED_CNN import PretrainedREDCNN
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from deeplib.history import History
from deeplib.training import HistoryCallback
#from deeplib.datasets import train_valid_loaders

from utils.util import get_preprocessing
from draw_images import draw_pred_target, draw_all_preds_targets
from model.metrics import validate
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
		save_path="model/entire_model",
		load_data_for_challenge=False,
		dataset_test_challenge=None,
		load_network_state=False,
		leonardo_dataset=None
):
	"""
	Entraîne un réseau de neurones PyTorch avec Poutyne. On suppose que la sortie du réseau est compatible avec
	la fonction cross-entropy de PyTorch pour calculer l'exactitude (accuracy).

	Args:
		network (nn.Module): Un réseau de neurones PyTorch
		optimizer (torch.optim.Optimizer): Un optimiseur PyTorch
		dataset (torch.utils.data.Dataset): Un jeu de données PyTorch
		n_epoch (int): Le nombre d'epochs d'entraînement désiré
		batch_size (int): La taille de batch désirée
		use_gpu (bool): Si on veut utiliser le GPU. Est vrai par défaut. Un avertissement est lancé s'il n'y a pas de
			GPU.
		criterion: Une fonction de perte compatible avec la cross-entropy de PyTorch.
		callbacks (List[poutyne.Callback]): Une liste de callbacks de Poutyne (utile pour les horaires d'entrainement
			entre autres).

	Returns:
		Retourne un objet de type `deeplib.history.History` contenant l'historique d'entraînement.
	"""
	history_callback = HistoryCallback()
	history = History()
	callbacks = [history_callback] if callbacks is None else [history_callback] + callbacks
	if leonardo_dataset is not None:
		train_loader, valid_loader = train_valid_loaders(dataset, batch_size=batch_size, valid_dataset=leonardo_dataset)

	else:
		train_loader, valid_loader = train_valid_loaders(dataset, batch_size=batch_size, train_split=0.9)


	# optimizer
	if optimizer == "Adam":
		opt = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
	elif optimizer == "SGD":
		opt = optim.SGD(network.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
	else:
		raise RuntimeError("{} optimizer not available!".format(optimizer))
	#scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)
	scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.94)


	# loss function
	if criterion == "MSELoss":
		loss = nn.MSELoss()
	else:
		raise RuntimeError("{}criterion not available!".format(optimizer))

	if use_gpu:
		network.cuda()

	n_batch = len(train_loader)
	t0 = time.time()
	if not load_network_state:
		best_RMSE = 1e9
		for i_epoch in range(n_epoch):
			network.train()
			with torch.enable_grad():
				for j, (inputs, targets) in enumerate(train_loader):
					if use_gpu:
						inputs = inputs.cuda()
						targets = targets.cuda()
					opt.zero_grad()
					output = network(inputs)
					batch_loss = loss(output, targets)
					batch_loss.backward()
					opt.step()
					j += 1
				# sys.stdout.write("%s[%s%s] %i/%i\r" % ("mini batch :", "#"*j, "."*(n_batch-j), j, n_batch))
				# sys.stdout.flush()
			train_loss, train_RMSE = validate(network, train_loader, loss, use_gpu=use_gpu)
			valid_loss, valid_RMSE = validate(network, valid_loader, loss, use_gpu=use_gpu)
			t1 = time.time()
			lr = opt.param_groups[0]['lr']
			if i_epoch % 2 == 0 and i_epoch != 0 and lr > 1e-5:
				scheduler.step()
			history.save(dict(acc=train_RMSE, val_acc=valid_RMSE, loss=train_loss, val_loss=valid_loss, lr=lr))
			print(f'Epoch {i_epoch} ({t1 - t0:.1f} s) - Train RMSE: {train_RMSE:.3e} - Val RMSE: {valid_RMSE:.3e} - Train loss: {train_loss:.3e} - Val loss: {valid_loss:.3e} - lr: {lr:.2e}')
			# --------------------------------------------------------------------------------- #
			#                            save best model parameters                             #
			# --------------------------------------------------------------------------------- #
			if valid_RMSE < best_RMSE:
				torch.save(network.state_dict(), "{}_best.pt".format(save_path))
				best_RMSE = valid_RMSE
		# --------------------------------------------------------------------------------- #
		#                            save model at the end                                  #
		# --------------------------------------------------------------------------------- #
		torch.save(network, "{}_end.pt".format(save_path))
		network = torch.load("{}_best.pt".format(save_path))
		network.eval()
		if use_gpu:
			network.cuda()
	else:
		network = torch.load("{}_best.pt".format(save_path))
		network.eval()
		if use_gpu:
			network.cuda()
	# --------------------------------------------------------------------------------- #
	#                            save challenge images                                  #
	# --------------------------------------------------------------------------------- #
	if dataset_test_challenge is not None:
		test_loader = DataLoader(dataset_test_challenge, batch_size=batch_size)
		validate(network, test_loader, loss, use_gpu=use_gpu, save_data=True, output_path="data/challenge")
		#draw_all_preds_targets(network, test_loader, os.path.relpath("../Figure_challenge"))
	# --------------------------------------------------------------------------------- #
	#                            save validation images                                 #
	# --------------------------------------------------------------------------------- #
	validate(network, valid_loader, loss, use_gpu=use_gpu, save_data=True)
	#draw_all_preds_targets(network, valid_loader)
	return history


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
	lr = 0.0002
	momentum = 0.9
	n_epoch = 100
	batch_size = 1
	weight_decay = 1e-4
	criterion = "MSELoss"
	optimizer = "Adam"

	# unet setup constants
	nb_filter=(64, 128, 256, 512, 1024)
	use_relu=False
	mode='nearest'
	residual_block=True
	if batch_size == 1:
		batch_norm_momentum = 0.01
	else:
		batch_norm_momentum = 0.05

	#seed
	seed = 42
	set_seed(seed)

	# --------------------------------------------------------------------------------- #
	#                            network                                                #
	# --------------------------------------------------------------------------------- #
	available_networks = [
		"UNet",
		"NestedUNet",
		"SMP UnetPLusPLus",
		"Pretrained Simple UNet",
		"Pretrained RED_CNN"
	]

	network_to_use: str = "UNet"

	if network_to_use not in available_networks:
		raise NotImplementedError(f"Chosen network isn't implemented \nImplemented networks are {available_networks}.")
	elif network_to_use is "UNet":
		model = UNet(1, 1,
					channels_depth_number=nb_filter,
					use_relu=use_relu,  # If False, then LeakyReLU
					mode=mode,  # For upsampling
					residual_block=residual_block,  # skip connections?
					batch_norm_momentum=batch_norm_momentum)
		preprocessing = None
	elif network_to_use is "NestedUNet":
		model = NestedUNet(1,1, nb_filter=nb_filter, batch_norm_momentum=batch_norm_momentum)
		preprocessing = None
	elif network_to_use is "SMP UnetPLusPLus":
		encoder = "densenet121"
		encoder_weights = "imagenet"
		activation = "logits"
		encoder_depth = 5
		decoder_channels = (1024, 512, 256, 128, 64)
		preprocessing = get_preprocessing(get_preprocessing_fn(encoder_name=encoder, pretrained=encoder_weights))
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
	elif network_to_use is "Pretrained Simple UNet":
		model = PretrainedUNet(
			1, unfreezed_layers=["up1", "up2", "up3", "up4", "up5", "outc"]
		)
		preprocessing = None
	elif network_to_use is "Pretrained RED_CNN":
		model = PretrainedREDCNN(unfreezed_layers=["conv", "tconv"])
		preprocessing = None
	else:
		warnings.warn("Something very wrong happened")

	logging.info(f"\nNombre de paramètres: {np.sum([p.numel() for p in model.parameters()])}")

	# --------------------------------------------------------------------------------- #
	#                            dataset                                                #
	# --------------------------------------------------------------------------------- #
	if load_data_for_challenge:
		train_images, test_images = load_all_images(image_types=["FBP", "Phantom", "Sinogram", "virtual_breast", "fdk"], n_batch=1,
			multiple_channels=False, load_sinograms=False, merge_datasets=False)
		valid_images_contest = load_images("FBP", path="data/validation", n_batch=1)
		aapm_dataset = BreastCTDataset(train_images["FBP"], train_images["PHANTOM"], preprocessing=preprocessing)
		aapm_dataset_valid = BreastCTDataset(valid_images_contest, valid_images_contest, preprocessing=preprocessing)
		leonardo_dataset = BreastCTDataset(train_images["FDK"][:100], train_images["VIRTUAL_BREAST"][:100], preprocessing=preprocessing)
	else:
		train_images, test_images = load_all_images(n_batch=1)
		aapm_dataset = BreastCTDataset(train_images["FBP"], train_images["PHANTOM"], preprocessing=preprocessing)
		leonardo_dataset = BreastCTDataset(train_images["FDK"], train_images["VIRTUAL_BREAST"], preprocessing=preprocessing)

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
		dataset_test_challenge=aapm_dataset_valid,
		load_network_state=load_network_state,
		save_path="model/model_state",
		leonardo_dataset=leonardo_dataset
	)

	# --------------------------------------------------------------------------------- #
	#                           network analysing                                       #
	# --------------------------------------------------------------------------------- #
	history.display()
