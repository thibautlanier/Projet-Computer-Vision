from torch.utils.data import Dataset
from torchvision.datasets import Flowers102
from typing import Any
from PIL import Image
import os

class CocoDataset(Dataset):
	def __init__(self, root:str, transforms = None, loader = None) -> None:
		super().__init__()
		self.dir = os.path.join(root, "test2017")
		if not self.is_present(self.dir):
			raise FileNotFoundError("Dataset not downloaded")
		self.data = os.listdir(self.dir)
		self.transforms = transforms
		if loader is not None:
			self.loader = loader
		else:
			self.loader = Image.open

	def is_present(self, dir:str) -> bool:
		return os.path.exists(dir)
	
	def __getitem__(self, index) -> Any:
		img = self.loader(self.data[index])
		if self.transforms is not None:
			img = self.transforms
		return img[0:1,...], img
	
	def __len__(self) -> int:
		return len(self.data)

class CustomFlowersDataset(Dataset):
	def __init__(self, root, split="train", transform=None, download=False):
		self.flowers_dataset = Flowers102(root=root, split=split, download=download)
		self.transform = transform

	def __getitem__(self, index):
		image, _ = self.flowers_dataset[index]
		if self.transform is not None:
			image = self.transform(image)
		return image[0:1,...], image

	def __len__(self):
		return len(self.flowers_dataset)