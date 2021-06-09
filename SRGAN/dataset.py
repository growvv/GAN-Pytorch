import os
import numpy as np
from numpy.lib.type_check import imag
from torch.utils import data
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyImageDataset(Dataset):
    def __init__(self, root_img_dir):
        super().__init__()
        self.root_img_dir = root_img_dir
        self.images = os.listdir(root_img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.root_img_dir, img_name)

        img = np.array(Image.open(img_path).convert("RGB"))

        img = config.both_transforms(image=img)["image"]
        lr_img = config.lowres_transform(image=img)["image"]
        hr_img = config.highres_transform(image=img)["image"]
        
        return lr_img, hr_img

    
def test():
    dataset = MyImageDataset(root_img_dir="data/hr/")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)
    print(len(loader))

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)

if __name__ == "__main__":
    test()
