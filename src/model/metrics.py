from sklearn.metrics import mean_squared_error
import torch
import numpy as np
import os
import pdb

from poutyne import Model
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# Classe de fonction de perte RMSE pour modele poutyne
class RMSELoss(nn.Module):
	def __init__(self, epsilon=1e-10):
		super().__init__()
		self.MSE = nn.MSELoss()
		self.epsilon = epsilon

	def forward(self, pred, target):
		loss = torch.sqrt(self.MSE(pred, target) + self.epsilon)
		return loss


class PSNRLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.PSNR = psnr

	def forward(self, pred, target):
		loss = self.PSNR(target.numpy()[0, 0, :, :], pred.numpy()[0, 0, :, :])
		return loss


class SSIMLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.SSIM = ssim

	def forward(self, pred, target):
		loss = self.SSIM(target.numpy()[0, 0, :, :], pred.numpy()[0, 0, :, :], multichannel=True)
		return loss


def contest_metric_evaluation(INPUT, OUT, evaluate_worst_RMSE=True):
	# INPUT which has both ./ref and ./res - user submission
	# OUT : OUTPUT

	REFERENCE = os.path.join(INPUT, "ref") # Phantom GT
	PREDICTION_OUTPUT = os.path.join(INPUT, "res") # user submission wll be available from here

	# Ground Truth
	# filtered back-projection reconstruction from the 128-view sinogram
	# users will try to recreate these. They serve as the input data.
	if len(os.listdir(REFERENCE)) == 1 and os.listdir(REFERENCE)[0][-4:] == ".npy":
		phantom_gt_file_name = os.listdir(REFERENCE)[0]
	else:
		raise Exception('Organizer, either you have more than one file in your ref directory or it doesn\'t end in .npy')

	# User Images
	# The goal is to train a network that accepts the FBP128 image (and/or the 128-view sinogram)
	# to yield an image that is as close as possible to the corresponding Phantom image.
	if len(os.listdir(PREDICTION_OUTPUT)) == 1 and os.listdir(PREDICTION_OUTPUT)[0][-4:] == ".npy":
		prediction_file_name = os.listdir(PREDICTION_OUTPUT)[0]
	else:
		raise Exception('You either have more than one file in your submission or it doesn\'t end in .npy')

	phantom_gt = np.load(os.path.join(REFERENCE, phantom_gt_file_name))
	prediction_phantoms = np.load(os.path.join(PREDICTION_OUTPUT,prediction_file_name))

	# get the number of prediction_phantoms and number of pixels in x and y
	nim, nx, ny = prediction_phantoms.shape

	# mean RMSE computation
	diffsquared = (phantom_gt-prediction_phantoms)**2
	num_pix = float(nx*ny)

	meanrmse  = np.sqrt( ((diffsquared/num_pix).sum(axis=2)).sum(axis=1) ).mean()
	print("The mean RMSE over %3i images is %8.6f "%(nim,meanrmse))

	# worst-case ROI RMSE computation
	roisize = 25  #width and height of test ROI in pixels
	x0 = 0        #tracks x-coordinate for the worst-case ROI
	y0 = 0        #tracks x-coordinate for the worst-case ROI
	im0 = 0       #tracks image index for the worst-case ROI

	maxerr = -1.
	if evaluate_worst_RMSE:
		for i in range(nim): # For each image
			print("Searching images: %3i/%3i"%(i+1,nim), end='\r')
			phantom = phantom_gt[i].copy() # GT
			prediction =  prediction_phantoms[i].copy() # Pred
			# These for loops cross every pixel in image (from region of interest)
			for ix in range(nx-roisize):
				for iy in range(ny-roisize):
					roiGT =  phantom[ix:ix+roisize,iy:iy+roisize].copy() # GT
					roiPred =  prediction[ix:ix+roisize,iy:iy+roisize].copy() # Pred
					if roiGT.max()>0.01: #Don't search ROIs in regions where the truth image is zero
						roirmse = np.sqrt( (((roiGT-roiPred)**2)/float(roisize**2)).sum() )
						if roirmse>maxerr:
							maxerr = roirmse
							x0 = ix
							y0 = iy
							im0 = i
		print("Worst-case ROI RMSE is %8.6f"%(maxerr))
		print("Worst-case ROI location is (%3i,%3i) in image number %3i "%(x0,y0,im0+1))

	with open(os.path.join(OUT,"scores.txt"), "w") as results:
		results.write("score_1: {}\n".format(meanrmse))
		results.write("score_2: {}".format(maxerr))

def validate(network, valid_loader, criterion, use_gpu=True, save_data=False,
		output_path="data/model_trained_results", pred_folder="res", target_folder="ref"):
	model = Model(network, optimizer=None, loss_function=criterion, batch_metrics=['accuracy'])
	if use_gpu:
		if torch.cuda.is_available():
			model.cuda()
		else:
			raise RuntimeError("No GPU available!")
	return validate_model(model, valid_loader, save_data=save_data, output_path=output_path,
		pred_folder=pred_folder, target_folder=target_folder)

def validate_model(model, valid_loader, save_data=False, output_path="data/model_trained_results",
		pred_folder="res", target_folder="ref", evaluate_worst_RMSE=True):
	results = model.evaluate_generator(
		valid_loader,
		return_pred=save_data,
		return_ground_truth=save_data,
		progress_options=dict(coloring=False))
	print(len(results))
	if save_data:
		if not os.path.isdir(os.path.join(output_path, pred_folder)):
			os.makedirs(os.path.join(output_path, pred_folder))
		if not os.path.isdir(os.path.join(output_path, target_folder)):
			os.makedirs(os.path.join(output_path, target_folder))
		np.save(os.path.join(output_path, pred_folder, "predictions"), np.squeeze(results[1]))
		np.save(os.path.join(output_path, target_folder, "targets"), np.squeeze(results[2]))
		contest_metric_evaluation(output_path, output_path, evaluate_worst_RMSE=evaluate_worst_RMSE)
	return results[0]


def validate_2(network, valid_loader, criterion, use_gpu=True, save_data=False, output_path="data/model_trained_results", pred_folder="res", target_folder="ref"):
	loss = []
	RMSE = []
	worst_RMSE = []
	store_preds = []
	store_targets = []
	network.eval()
	if not use_gpu:
		network.cpu()
	with torch.no_grad():
		for i, (inputs, targets) in enumerate(valid_loader):
			if use_gpu:
				inputs = inputs.cuda()
				targets = targets.cuda()
			pred = network(inputs)
			if type(pred) is list:
				pred = pred[0]
			tmp_loss = criterion(pred, targets)
			loss.append(tmp_loss)
			targets = targets.cpu().numpy()
			pred = pred.cpu().numpy()
			# for i in range(targets.shape[0]):
			# 	RMSE.append(np.sqrt((((targets[i][0] - pred[i][0])**2/(pred.shape[2] * pred.shape[3])).sum(axis=1)).sum(axis=0)).mean())
				#RMSE.append(mean_squared_error(targets[i][0], pred[i][0], squared=False))
			if save_data:
				store_preds.append(pred.reshape(pred.shape[0], pred.shape[2], pred.shape[3]))
				if targets is not None:
					store_targets.append(targets.reshape(targets.shape[0], targets.shape[2], targets.shape[3]))
		loss = torch.tensor(loss)
	if save_data:
		if not os.path.isdir(os.path.join(output_path, pred_folder)):
			os.makedirs(os.path.join(output_path, pred_folder))
		if not os.path.isdir(os.path.join(output_path, target_folder)):
			os.makedirs(os.path.join(output_path, target_folder))
		store_preds = np.concatenate(tuple(store_preds))
		np.save(os.path.join(output_path, pred_folder, "predictions"), store_preds)
		if len(store_targets) != 0:
			store_targets = np.concatenate(tuple(store_targets))
			np.save(os.path.join(output_path, target_folder, "targets"), store_targets)
		contest_metric_evaluation(output_path, output_path)
	return torch.mean(loss)
