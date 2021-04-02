from sklearn.metrics import mean_squared_error
import torch
import numpy as np
import os
import pdb

def contest_metric_evaluation(INPUT, OUT):
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
	print("The mean RSME over %3i images is %8.6f "%(nim,meanrmse))

	# worst-case ROI RMSE computation
	roisize = 25  #width and height of test ROI in pixels
	x0 = 0        #tracks x-coordinate for the worst-case ROI
	y0 = 0        #tracks x-coordinate for the worst-case ROI
	im0 = 0       #tracks image index for the worst-case ROI

	maxerr = -1.
	for i in range(nim): # For each image
	   print("Searching image %3i"%(i))
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

def validate(network, valid_loader, criterion, use_gpu=True, save_data=False, output_path="data/model_trained_results", pred_folder="res", target_folder="ref"):
	loss = []
	RMSE = []
	worst_RMSE = []
	store_preds = []
	store_targets = []
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
				RMSE.append(np.sqrt((((targets[i][0] - pred[i][0])**2/(pred.shape[2] * pred.shape[3])).sum(axis=1)).sum(axis=0)).mean())
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
	return torch.mean(loss), np.mean(RMSE)

