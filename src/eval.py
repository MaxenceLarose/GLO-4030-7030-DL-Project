# eval.py
import os

import torch.nn as nn
import torch

from data_loader.data_loaders import load_all_images
from data_loader.datasets import BreastCTDataset
from model.metrics import validate, RMSELoss

def eval_model(model_file, batch_size=1, use_gpu=True):
	# --------------------------------------------------------------------------------- #
	#                            network                                                #
	# --------------------------------------------------------------------------------- #
	model = torch.load_state_dict(torch.load(model_file))
	model.eval()
	if use_gpu:
		model.cuda()

	# --------------------------------------------------------------------------------- #
	#                            dataset                                                #
	# --------------------------------------------------------------------------------- #
	train_images, test_images = load_all_images(n_batch=4)
	dataset = BreastCTDataset(train_images["FBP"], train_images["PHANTOM"], preprocessing=None)
	train_loader, valid_loader = train_valid_loaders(dataset, batch_size=batch_size, train_split=0.9)

	# --------------------------------------------------------------------------------- #
	#                            eval                                                   #
	# --------------------------------------------------------------------------------- #
	loss = RMSELoss
	validate(model, valid_loader, loss, use_gpu=use_gpu, save_data=False)


if __name__ == '__main__':
	eval_model(os.path.relpath("../training/resnet34_unetpp_train_encoder_deoder_512/model_state_best.pt"))
