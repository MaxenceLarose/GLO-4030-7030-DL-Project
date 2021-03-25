from sklearn.metrics import mean_squared_error
import torch
import numpy as np


def validate(network, valid_loader, criterion, use_gpu=True):
	loss = []
	RMSE = []
	worst_RMSE = []
	network.eval()
	if not use_gpu:
		network.cpu()
	with torch.no_grad():
		for inputs, targets in valid_loader:
			if use_gpu:
				inputs = inputs.cuda()
				targets = targets.cuda()
			pred = network(inputs)
			loss.append(criterion(pred, targets))
			targets = targets.cpu().numpy()
			pred = pred.cpu().numpy()
			for i in range(targets.shape[0]):
				RMSE.append(mean_squared_error(targets[i][0], pred[i][0], squared=False))
		loss = torch.tensor(loss)
	return torch.mean(loss), np.mean(RMSE)
