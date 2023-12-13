import numpy as np
import cv2
import torch

def compute_psnr(img1, img2):
	mse = np.mean((img1 - img2) ** 2)
	if mse == 0:
		return float('inf')
	max_pixel = 255.0
	psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
	# psnr = cv2.PSNR(img1, img2)
	return psnr

def compute_ssim(img1, img2):
	ssim = cv2.SSIM(img1, img2)
	return ssim

def compute_mse(img1, img2):
	mse = np.mean((img1 - img2) ** 2)
	return mse

def batch_metrics(model, X, y, device, metric="PSNR"):
	batch_size = X.shape[0]
	model = model.to(device)
	model.eval()
	X = X.to(device)
	with torch.no_grad():
		output = model(X)
		output = output.cpu().numpy()
		X = X.cpu().numpy()
		y = y.cpu().numpy()
	res = 0
	m = compute_psnr
	if metric == "SSIM":
		m = compute_ssim
	elif metric == "MSE":
		m = compute_mse
	for i in range(batch_size):
		res += m(y[i], output[i])
	return res