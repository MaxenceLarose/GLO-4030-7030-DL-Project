from datasets import load_all_images, BreastCTDataset
from deeplib.history import History
from deeplib.training import test, HistoryCallback, get_model
from deeplib.datasets import train_valid_loaders
import poutyne as pt
import torch.optim as optim
import torch.nn as nn
import torch
from unet import UNet
from draw_images import draw_data_targets, draw_pred_target
import numpy as np
from sklearn.metrics import mean_squared_error
from metrics import validate
import sys
import time


def get_model(network, epoch_metrics, optimizer=None, criterion=None, use_gpu=True):
    """
    Obtient un modèle Poutyne pour un réseau de neurones PyTorch. On suppose que la sortie du réseau est compatible avec
    la fonction cross-entropy de PyTorch pour pouvoir utiliser l'exactitude (accuracy).

    Args:
        network (nn.Module): Un réseau de neurones PyTorch
        optimizer (torch.optim.Optimizer): Un optimiseur PyTorch
        criterion: Une fonction de perte compatible avec la cross-entropy de PyTorch
        use_gpu (bool): Si on veut utiliser le GPU. Est vrai par défaut. Un avertissement est lancé s'il n'y a pas de
            GPU.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model = pt.Model(network, optimizer, criterion, epoch_metrics=epoch_metrics)
    if use_gpu:
        if torch.cuda.is_available():
            model.cuda()
        else:
            warnings.warn("Aucun GPU disponible")
    return model

# Fonction qui provient de la librairie deeplib (https://github.com/ulaval-damas/glo4030-labs/tree/master/deeplib)
# Le code a été adapté pour utiliser toutes les données mnist pour l'entrainement.
def train_network(network, dataset, *, optimizer : str="Adam", lr=0.001, weight_decay=0, n_epoch=5, batch_size=1, use_gpu=True, 
	criterion="MSELoss", callbacks=None, seed=42):
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
	train_loader, valid_loader = train_valid_loaders(dataset, batch_size=batch_size)
	# optimizer
	if optimizer == "Adam":
		opt = optim.Adam(unet.parameters(), lr=lr, weight_decay=weight_decay)
	elif optimizer == "SGD":
		opt = optim.SGD(unet.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
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
		print(f'Epoch {i_epoch} ({t1-t0:.1f} s) - Train RMSE: {train_RMSE:.8f} - Val RMSE: {valid_RMSE:.8f} - Train loss: {train_loss:.8f} - Val loss: {valid_loss:.8f} - lr: {lr:.2e}')

	#batch_metrics = pt.SKLearnMetrics([mean_squared_error])
	# epoch_metrics = pt.SKLearnMetrics([mean_squared_error])
	# model = get_model(network, [epoch_metrics], opt, loss, use_gpu=use_gpu)
	# model.fit_generator(train_loader,
	# 					valid_loader,
	# 					epochs=n_epoch,
	# 					progress_options=dict(coloring=False),
	# 					callbacks=callbacks)
	image_idx = 0
	with torch.no_grad():
		for i, (inputs, targets) in enumerate(valid_loader):
			if use_gpu:
				inputs = inputs.cuda()
				targets = targets.cuda()
			pred = network(inputs)
			pred = pred.cpu().numpy()
			inputs = inputs.cpu().numpy()
			targets = targets.cpu().numpy()
			draw_pred_target(inputs, targets, pred, image_idx=image_idx, fig_id=i)
	return history

#def train_network_2


if __name__ == '__main__':
	train_images, test_images = load_all_images(n_batch=4)
	breast_CT_dataset_train = BreastCTDataset(train_images["FBP"], train_images["PHANTOM"])
	#draw_data_targets(breast_CT_dataset_train)
	#exit(0)
	unet = UNet(1,1)
	print("Nombre de paramètres:", np.sum([p.numel() for p in unet.parameters()]))

	lr = 0.001
	n_epoch = 100
	batch_size = 1
	weight_decay = 1e-4
	criterion = "MSELoss"
	optimizer = "SGD"
	history = train_network(unet, breast_CT_dataset_train, optimizer=optimizer, lr=lr, weight_decay=weight_decay,
	 n_epoch=n_epoch, batch_size=batch_size, criterion=criterion, use_gpu=True)
	history.display()

