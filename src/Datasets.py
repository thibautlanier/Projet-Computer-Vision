from torch import zeros_like, cat
from torch.utils.data import Dataset
from torchvision.datasets import Flowers102
from typing import Any
from PIL import Image
from random import randint
import os

SIZE = 128  
def choose_pixel():
    return (randint(0, SIZE-1), randint(0, SIZE-1))

def ab_percent(percent=0.01):
    return int(SIZE**2 * percent)


class CocoDataset(Dataset):
	def __init__(self, root:str, split:str = "train", add_pixels:bool = False, transform:callable = None, loader:callable = None) -> None:
		super().__init__()
		self.dir = os.path.join(root, "coco_dataset", split)
		if not self.is_present(self.dir):
			raise FileNotFoundError("Dataset not downloaded")
		self.data = os.listdir(self.dir)
		self.transforms = transform
		if loader is not None:
			self.loader = loader
		else:
			self.loader = Image.open

	def is_present(self, dir:str) -> bool:
		return os.path.exists(dir)
	
	def get_info(self) -> str:
		return f"Dataset: {self.dir} \tNumber of images: {len(self.data)}"

	def __getitem__(self, index) -> Any:
		img = self.loader(os.path.join(self.dir, self.data[index])).convert("RGB")
		if self.transforms is not None:
			img = self.transforms(img)
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


        l_channel = image[0, ...].unsqueeze(0)
        a_channel = image[1, ...].unsqueeze(0)
        b_channel = image[2, ...].unsqueeze(0)

        modified_a_channel = zeros_like(a_channel)
        modified_b_channel = zeros_like(b_channel)

        num_pixels_to_modify = ab_percent()
        
        for i in range(num_pixels_to_modify):
            chosen_pixel = choose_pixel()
            chosen_pixel_color = image[:, chosen_pixel[0], chosen_pixel[1]].unsqueeze(1).unsqueeze(2)
            modified_a_channel[:, chosen_pixel[0], chosen_pixel[1]] = chosen_pixel_color[1, 0, 0]
            modified_b_channel[:, chosen_pixel[0], chosen_pixel[1]] = chosen_pixel_color[2, 0, 0]

        lab_image = cat([l_channel, modified_a_channel, modified_b_channel], dim=0)

        return lab_image, image[1:, ...]

    def __len__(self):
        return len(self.flowers_dataset)