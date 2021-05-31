import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader, dataset
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator

def train_fn(disc_H, disc_Z, gen_H, gen_Z, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (horse, zebra) in enumerate(loop):
        horse = horse.to(config.DEVICE)
        zebra = zebra.to(config.DEVICE)

        # 训练判别器 H Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            D_H_real_loss = MSE(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = MSE(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = MSE(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = MSE(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # 训练生成器 H Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = MSE(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = MSE(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = L1(zebra, cycle_zebra)
            cycle_horse_loss = L1(horse, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            # 防止生成一个一样的？？
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = L1(zebra, identity_zebra)
            identity_horse_loss = L1(horse, identity_horse)

            G_loss = (   # 6个loss 
                loss_G_H
                + loss_G_Z
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
                + identity_zebra_loss * config.LAMBDA_IDENTITY
                + identity_horse_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 1 == 0:
            save_image(fake_horse*0.5+0.5, f"saved_images/fake_horse_{idx}.png")
            save_image(fake_zebra*0.5+0.5, f"saved_images/fake_zebra_{idx}.png")
            save_image(horse*0.5+0.5, f"saved_images/horse_{idx}.png")
            save_image(zebra*0.5+0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(H_real=D_H_real_loss/(idx+1), H_fake=D_H_fake_loss/(idx+1))


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3).to(config.DEVICE)
    gen_H = Generator(img_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_H, disc_H, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_Z, disc_Z, opt_disc, config.LEARNING_RATE)

    train_dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR+"/horses", root_zebra=config.TRAIN_DIR+"/zebras", transform=config.transforms
    )
    val_dataset = HorseZebraDataset(
        root_horse=config.VAL_DIR+"/horses", root_zebra=config.VAL_DIR+"/zebras", transform=config.transforms
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_H, disc_Z, gen_H, gen_Z, train_loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_DISC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_DISC_Z)

if __name__ == "__main__":
    main()


