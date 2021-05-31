from PIL import Image
import os
from albumentations import augmentations
from torch.utils.data import Dataset
import numpy as np

class HorseZebraDataset(Dataset):
    def __init__(self, root_horse, root_zebra, transform=None):
        super().__init__()
        self.root_horse = root_horse
        self.root_zebra = root_zebra
        self.transform = transform

        self.horse_images = os.listdir(root_horse)
        self.zebra_images = os.listdir(root_zebra)
        self.length_dataset = max(len(self.horse_images), len(self.zebra_images))
        self.horse_len = len(self.horse_images)
        self.zebra_len = len(self.zebra_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        horse_img_name = self.horse_images[index % self.horse_len]
        zebra_img_name = self.zebra_images[index % self.zebra_len]
        horse_path = os.path.join(self.root_horse, horse_img_name)
        zebra_path = os.path.join(self.root_zebra, zebra_img_name)
        horse_img = np.array(Image.open(horse_path).convert("RGB"))
        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image0=horse_img, image=zebra_img)
            horse_img = augmentations["image0"]
            zebra_img = augmentations["image"]

        return horse_img, zebra_img


