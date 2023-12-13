import torch
from torch import nn, cat, relu, save, load

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2, padding_mode="replicate"),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2, padding_mode="replicate"),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.t_conv4 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

        self.output = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1, padding_mode="replicate")

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        x_4 = self.conv4(x_3)
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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(0.2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))

        self.output = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, padding_mode="replicate")

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        x_4 = self.conv4(x_3)
        return torch.sigmoid(self.output(x_4))

def save_model(model, path=None):
    if path is None:
        path = f"{model.__class__.__name__}.pt"
    save(model.state_dict(), path)

def load_model(model, path=None):
    if path is None:
        path = f"{model.__class__.__name__}.pt"
    model.load_state_dict(load(path))
    return model