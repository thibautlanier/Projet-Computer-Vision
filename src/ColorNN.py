import torch
from torch import nn, concat

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
			nn.Conv2d(c_out, c_out, k_size, stride, padding, padding_mode=padding_mode),
			nn.ReLU()
		)
	
class UpscaleResidualNN(ColorNN):
	def __init__(self) -> None:
		super().__init__()
		
		self.encod1 = nn.Sequential(
			self.DoubleConv2d(1, 16),
			nn.BatchNorm2d(16)
		)
		self.encod2 = nn.Sequential(
			self.DoubleConv2d(16, 32),
			nn.BatchNorm2d(32)
		)
		self.encod3 = nn.Sequential(
			self.DoubleConv2d(32, 64),
			nn.BatchNorm2d(64)
		)
		self.encod4 = nn.Sequential(
			self.DoubleConv2d(64, 128),
			nn.BatchNorm2d(128)
		)

		self.decod1 = nn.Sequential(
			nn.ConvTranspose2d(128, 64, 2, 2),
			nn.BatchNorm2d(64)
		)
		self.decod2 = nn.Sequential(
			self.DoubleConv2d(128, 64),
			nn.ConvTranspose2d(64, 32, 2, 2),
			nn.BatchNorm2d(32)
		)
		self.decod3 = nn.Sequential(
			self.DoubleConv2d(64, 32),
			nn.ConvTranspose2d(32, 16, 2, 2),
			nn.BatchNorm2d(16)
		)
		self.decod4 = nn.Sequential(
			self.DoubleConv2d(32, 16),
			nn.Conv2d(16, 2, 1),
			nn.BatchNorm2d(2)
		)

	
	def forward(self, X):
		X_1 = self.encod1(X) #16
		X_2 = self.encod2(nn.MaxPool2d(2)(X_1)) #32
		X_4 = self.encod3(nn.MaxPool2d(2)(X_2)) #64
		X_8 = self.encod4(nn.MaxPool2d(2)(X_4)) #128

		result = self.decod1(X_8)
		result = self.decod2(concat((X_4,result), dim=1))
		result = self.decod3(concat((X_2,result), dim=1))
		result = self.decod4(concat((X_1,result), dim=1))
		return concat((X, result), dim=1)
