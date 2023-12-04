import torch
from torch import nn

class ColorNN(nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.conv1 = self.DoubleConv2d(1, 3)
		self.conv2 = self.DoubleConv2d(3, 64)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, 3, 2, 1, padding_mode="replicate"),
			nn.ReLU()
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(128, 256, 3, 2, 1, padding_mode="replicate"),
			nn.ReLU()
		)

		self.conv5 = self.DoubleConv2d(451, 128)
		self.conv6 = self.DoubleConv2d(128, 16)
	
		self.conv7 = self.DoubleConv2d(144, 32)
		self.conv8 = self.DoubleConv2d(32, 2)

	def forward(self, X):
		
		X_1 = self.conv1(X)
		X_2 = self.conv2(X_1)
		X_3 = self.conv3(X_2)
		X_4 = self.conv4(X_3)
		X_5 = self.conv5(torch.concat([X_1,X_2,nn.Upsample(scale_factor=2)(X_3),nn.Upsample(scale_factor=4)(X_4)],dim = 1))
		X_6 = self.conv6(X_5)

		return torch.concat((X, self.conv8(self.conv7(torch.concat((X_5,X_6), dim=1)))), dim = 1)
		
	
	def DoubleConv2d(self, c_in, c_out, k_size=3, stride=1, padding=1, padding_mode="replicate"):
		return nn.Sequential(
			nn.Conv2d(c_in, c_out, k_size, stride, padding, padding_mode=padding_mode),
			nn.ReLU(),
			# nn.Conv2d(c_out, c_out, k_size, stride, padding, padding_mode=padding_mode),
			# nn.ReLU()
		)