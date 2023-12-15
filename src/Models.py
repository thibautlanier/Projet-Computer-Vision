from torch import nn, cat, relu, save, load

class NetworkColor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),)


        # Dilation layers.
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),)
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),)

        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),)
        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),)
        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),)
        self.t_conv4 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

        self.output = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Implements the forward pass for the given data `x`.
        :param x: The input data.
        :return: The neural network output.
        """
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        x_4 = self.conv4(x_3)

        # Dilation layers.
        x_5 = self.conv5(x_4)
        x_5_d = self.conv6(x_5)

        x_6 = self.t_conv1(x_5_d)
        x_6 = cat((x_6, x_3), 1)
        x_7 = self.t_conv2(x_6)
        x_7 = cat((x_7, x_2), 1)
        x_8 = self.t_conv3(x_7)
        x_8 = cat((x_8, x_1), 1)
        x_9 = self.t_conv4(x_8)
        x_9 = cat((x_9, x), 1)
        return cat((x, self.output(x_9)), 1)
    
class NetworkColorWithScribble(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)

        # Dilation layers.
        self.conv5 = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.conv6_bn = nn.BatchNorm2d(256)

        self.t_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.t_conv1_bn = nn.BatchNorm2d(128)
        self.t_conv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.t_conv2_bn = nn.BatchNorm2d(64)
        self.t_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.t_conv3_bn = nn.BatchNorm2d(32)
        self.t_conv4 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

        self.output = nn.Conv2d(5, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Implements the forward pass for the given data `x`.
        :param x: The input data.
        :return: The neural network output.
        """
        x_1 = relu(self.conv1_bn(self.conv1(x)))
        x_2 = relu(self.conv2_bn(self.conv2(x_1)))
        x_3 = relu(self.conv3_bn(self.conv3(x_2)))
        x_4 = relu(self.conv4_bn(self.conv4(x_3)))

        # Dilation layers.
        x_5 = relu(self.conv5_bn(self.conv5(x_4)))
        x_5_d = relu(self.conv6_bn(self.conv6(x_5)))

        x_6 = relu(self.t_conv1_bn(self.t_conv1(x_5_d)))
        x_6 = cat((x_6, x_3), 1)
        x_7 = relu(self.t_conv2_bn(self.t_conv2(x_6)))
        x_7 = cat((x_7, x_2), 1)
        x_8 = relu(self.t_conv3_bn(self.t_conv3(x_7)))
        x_8 = cat((x_8, x_1), 1)
        x_9 = relu(self.t_conv4(x_8))
        x_9 = cat((x_9, x), 1)
        x_f = self.output(x_9)

        return cat((x[:,0,...].unsqueeze(1), x_f), 1)

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
		X_5 = self.conv5(cat([X_1,X_2,nn.Upsample(scale_factor=2)(X_3),nn.Upsample(scale_factor=4)(X_4)],dim = 1))
		X_6 = self.conv6(X_5)

		return cat((X, self.conv8(self.conv7(cat((X_5,X_6), dim=1)))), dim = 1)
		
	
	def DoubleConv2d(self, c_in, c_out, k_size=3, stride=1, padding=1, padding_mode="replicate"):
		return nn.Sequential(
			nn.Conv2d(c_in, c_out, k_size, stride, padding, padding_mode=padding_mode),
			nn.ReLU(),
			nn.Conv2d(c_out, c_out, k_size, stride, padding, padding_mode=padding_mode),
			nn.ReLU()
		)

def save_model(model, path=None):
    if path is None:
        path = f"{model.__class__.__name__}.pt"
    save(model.state_dict(), path)

def load_model(model, path=None):
    if path is None:
        path = f"{model.__class__.__name__}.pt"
    model.load_state_dict(load(path))
    return model