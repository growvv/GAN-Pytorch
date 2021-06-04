from os import makedirs
from PIL.Image import Image, TRANSPOSE
import torch
from torch.autograd import grad_mode
from torch.serialization import load
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import config

def save_checkpoint(state, filename=config.CHECKPOINT_PTH):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    ):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
    )
    val_ds = CarvanaDataset(
        image_dir = val_dir,
        mask_dir = val_mask_dir,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds*y).sum()) / ((preds+y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="save_images/"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        save_image(preds, f"{folder}pred_{idx}.png")
        save_image(y.unsqueeze(1), f"{folder}_input{idx}.png")

    model.train()
