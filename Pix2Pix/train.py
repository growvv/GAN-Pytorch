from sys import setprofile
from typing import ItemsView
from albumentations.augmentations.functional import pad_with_params, scale
from numpy import index_exp
import torch
from torch import nn
from torch.nn.modules.activation import Sigmoid
from torch.serialization import load, save
from utils import load_checkpoint, save_image, save_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm  # 好看的进度条
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, val_loader):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # 训练识别器
        # with torch.cuda.amp.autocast():  # 自动混合精度
        y_fake = gen(x)
        d_real = disc(x, y)
        d_real_loss = bce(d_real, torch.ones_like(d_real))
        d_fake = disc(x, y_fake.detach())
        d_fake_loss = bce(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_real_loss + d_fake_loss) / 2

        disc.zero_grad()
        # d_scaler.scale(d_loss).backward()  # 没有cuda 白忙活
        # d_scaler.step(opt_disc)
        # d_scaler.update()
        d_loss.backward()
        opt_disc.step()

        # 训练生成器
        # with torch.cuda.amp.autocast():
        g_fake = disc(x, y_fake)
        g_fake_loss = bce(g_fake, torch.ones_like(d_fake))
        L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
        g_loss = g_fake_loss + L1

        gen.zero_grad()
        # g_scaler.scale(g_loss).backward()
        # g_scaler.step(opt_gen)
        # g_scaler.update()
        g_loss.backward()
        opt_gen.step()

        if idx % 10 == 0:
            loop.set_postfix(
                d_real = torch.sigmoid(d_real).mean().item(),
                d_fake = torch.sigmoid(d_fake).mean().item(),
            )

        if idx % 1 == 0:
            print(
                f'Batch {idx}/{len(loop)} Loss D: {d_loss}, loss G: {g_loss}'
            ) 

            save_some_examples(gen, val_loader, epoch=1, idx=idx, folder="evaluation")


def  main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_loss = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC,  disc, opt_disc, config.LEARNING_RATE)

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    # g_scaler = torch.cuda.amp.GradScaler()
    # d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) 

    for epoch in range(config.NUM_EPOCHS):
        print("epoch", epoch)
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_loss, BCE, val_loader)
        print("epoch after")

        save_some_examples(gen, val_loader, epoch, folder="evaluation")

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()



