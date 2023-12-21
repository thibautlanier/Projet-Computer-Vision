import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from cv2 import color
import torch

def tensorToImg(tensor:torch.Tensor):
	if len(tensor.shape) == 4:
		tensor = tensor.squeeze(0)
	return tensor.moveaxis(0,2).detach().cpu().numpy()


def compute_psnr(img1, img2):
	# mse = np.mean((img1 - img2) ** 2)
	# if mse == 0:
	# 	return float('inf')
	# max_pixel = 255.0
	# psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
	p = psnr(img1, img2, data_range=(img2.max() - img2.min()))
	return p

def compute_ssim(img1, img2):
	s = ssim(img1, img2, channel_axis=0, data_range=(img2.max() - img2.min()))
	return s

def batch_metrics(model, X, y, device, metric="PSNR"):
	batch_size = len(X)
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
	for i in range(batch_size):
		res += m(y[i], output[i])
	return res