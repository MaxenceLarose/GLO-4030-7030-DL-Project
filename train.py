from datasets import load_all_images, BreastCTDataset
from deeplib.history import History
from deeplib.training import test, HistoryCallback, get_model
from deeplib.datasets import train_valid_loaders
import poutyne as pt
import torch.optim as optim
from unet import UNet
import torch.nn as nn
from draw_images import draw_data_targets
import numpy as np

# Fonction qui provient de la librairie deeplib (https://github.com/ulaval-damas/glo4030-labs/tree/master/deeplib)
# Le code a été adapté pour utiliser toutes les données mnist pour l'entrainement.
def train_network(network, optimizer, dataset, n_epoch, batch_size, *, use_gpu=True, criterion=None, callbacks=None, seed=42):
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
	callbacks = [history_callback] if callbacks is None else [history_callback] + callbacks

	train_loader, valid_loader = train_valid_loaders(dataset, batch_size=batch_size)
	print(len(train_loader))

	model = get_model(network, optimizer, criterion, use_gpu=use_gpu)
	model.fit_generator(train_loader,
						valid_loader,
						epochs=n_epoch,
						progress_options=dict(coloring=False),
						callbacks=callbacks)

	return history_callback.history


if __name__ == '__main__':
	train_images, test_images = load_all_images(n_batch=1)
	breast_CT_dataset_train = BreastCTDataset(train_images["FBP"], train_images["PHANTOM"])
	#draw_data_targets(breast_CT_dataset_train)
	#exit(0)
	unet = UNet(1,1)
	print("Nombre de paramètres:", np.sum([p.numel() for p in unet.parameters()]))

	lr = 0.1
	n_epoch = 1
	batch_size = 16
	criterion = nn.MSELoss()
	optimizer = optim.Adam(unet.parameters(), lr=lr)
	train_network(unet, optimizer, breast_CT_dataset_train, n_epoch, batch_size, criterion=criterion)

