import logging
import time
import numpy as np

import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn as nn
import torch
from model.unet import UNet

from deeplib.history import History
from deeplib.training import HistoryCallback
from deeplib.datasets import train_valid_loaders

from draw_images import draw_pred_target
from model.metrics import validate
from data_loader.data_loaders import load_all_images
from data_loader.datasets import BreastCTDataset
from logger.logging_tools import logs_file_setup, log_device_setup, set_seed


# from segmentation_models_pytorch import UnetPlusPlus
# from segmentation_models_pytorch import Unet as smp_Unet
# from segmentation_models_pytorch.encoders import get_preprocessing_fn


# Fonction qui provient de la librairie deeplib (https://github.com/ulaval-damas/glo4030-labs/tree/master/deeplib)
# Le code a été adapté pour utiliser toutes les données mnist pour l'entrainement.
def train_network(
		network: nn.Module,
		dataset,
		*,
		optimizer: str = "Adam",
		lr=0.001,
		momentum=0.9,
		weight_decay=0,
		n_epoch=5,
		batch_size=1,
		use_gpu=True,
		criterion="MSELoss",
		callbacks=None
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
	train_loader, valid_loader = train_valid_loaders(dataset, batch_size=batch_size, train_split=0.9)

	# optimizer
	if optimizer == "Adam":
		opt = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
	elif optimizer == "SGD":
		opt = optim.SGD(network.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
	else:
		raise RuntimeError("{} optimizer not available!".format(optimizer))
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)

	# loss function
	if criterion == "MSELoss":
		loss = nn.MSELoss()
	else:
		raise RuntimeError("{}criterion not available!".format(optimizer))

	if use_gpu:
		network.cuda()

	n_batch = len(train_loader)
	t0 = time.time()
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
		scheduler.step(valid_loss)
		lr = opt.param_groups[0]['lr']
		history.save(dict(acc=train_RMSE, val_acc=valid_RMSE, loss=train_loss, val_loss=valid_loss, lr=lr))
		print(f'Epoch {i_epoch} ({t1 - t0:.1f} s) - Train RMSE: {train_RMSE:.8f} - Val RMSE: {valid_RMSE:.8f} - Train loss: {train_loss:.8f} - Val loss: {valid_loss:.8f} - lr: {lr:.2e}')
	# --------------------------------------------------------------------------------- #
	#                            save validation images                                 #
	# --------------------------------------------------------------------------------- #
	image_idx = 0
	valid_mssim = []
	valid_rmse = []
	with torch.no_grad():
		for i, (inputs, targets) in enumerate(valid_loader):
			print(i)
			if use_gpu:
				inputs = inputs.cuda()
				targets = targets.cuda()
			pred = network(inputs)
			pred = pred.cpu().numpy()
			inputs = inputs.cpu().numpy()
			targets = targets.cpu().numpy()
			mssim, rmse = draw_pred_target(inputs, targets, pred, image_idx=image_idx, fig_id=i)
			valid_mssim.append(mssim)
			valid_rmse.append(rmse)

	fig, ax = plt.subplots(1,2)
	ax[0].hist(valid_mssim, bins=20)
	ax[0].set_title("SSIM")
	ax[1].hist(valid_rmse, bins=20)
	ax[1].set_title("RMSE")
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
	lr = 0.0001
	momentum = 0.9
	n_epoch = 50
	batch_size = 1
	weight_decay = 0
	criterion = "MSELoss"
	optimizer = "Adam"
	batch_norm_momentum = 0.1
	seed = 42

	set_seed(seed)

	# --------------------------------------------------------------------------------- #
	#                            dataset                                                #
	# --------------------------------------------------------------------------------- #
	train_images, test_images = load_all_images(n_batch=4)
	breast_CT_dataset_train = BreastCTDataset(train_images["FBP"], train_images["PHANTOM"])
	# draw_data_targets(breast_CT_dataset_train)
	# exit(0)
	unet = UNet(1, 1, 
		channels_depth_number=(64, 128, 256, 512, 1024),
		use_relu=False, # If False, then LeakyReLU 
		mode='nearest', # For upsampling
		residual_block=True, # skip connections?
		batch_norm_momentum=batch_norm_momentum)
	logging.info(f"\nNombre de paramètres: {np.sum([p.numel() for p in unet.parameters()])}")

	# unet_plus_plus = UnetPlusPlus(in_channels=1, decoder_channels=(256, 128, 64, 32, 16), encoder_depth=5, encoder_weights='imagenet')
	# logging.info(f"\nNombre de paramètres: {np.sum([p.numel() for p in unet_plus_plus.parameters()])}")
	# preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
	# --------------------------------------------------------------------------------- #
	#                           network training                                        #
	# --------------------------------------------------------------------------------- #
	history = train_network(
		unet,
		breast_CT_dataset_train,
		optimizer=optimizer,
		lr=lr,
		momentum=momentum,
		weight_decay=weight_decay,
		n_epoch=n_epoch,
		batch_size=batch_size,
		criterion=criterion,
		use_gpu=True
	)

	# --------------------------------------------------------------------------------- #
	#                           network analysing                                       #
	# --------------------------------------------------------------------------------- #
	history.display()
