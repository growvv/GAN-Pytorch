import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules import fold
import torch.optim as optim
from torch.serialization import save
from tqdm import tqdm
import config
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

def train_fn(loader, model, optimizer, BCE, scaler, val_loader):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.float().unsqueeze(1).to(device=config.DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = BCE(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        save_predictions_as_imgs(val_loader, model, folder=config.SAVE_IMAGES)


def main():
    model = UNET(in_channels=3, out_channels=1).to(config.DEVICE)
    BCE = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        train_dir=config.TRAIN_IMG_DIR,
        train_mask_dir=config.TRAIN_MASK_DIR,
        val_dir=config.VAL_IMG_DIR,
        val_mask_dir=config.VAL_MASK_DIR,
        batch_size=config.BATCH_SIZE,
        train_transform=config.train_transform,
        val_transform=config.val_transform,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    if config.LOAD_MODEL:
        load_checkpoint(torch.load(config.CHECKPOINT_PTH), model)

    check_accuracy(val_loader, model)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, BCE, scaler, val_loader)

        # save model
        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint) 

        # check accuracy
        check_accuracy(val_loader, model)

        # print some example
        save_predictions_as_imgs(val_loader, model, folder=config.SAVE_IMAGES)

if __name__ == "__main__":
    main()

