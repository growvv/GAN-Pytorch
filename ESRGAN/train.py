from os import write
from numpy import generic
import torch
from torch.nn.modules.activation import LeakyReLU
from torch.serialization import load
import config
from torch import nn
from torch import optim
from utils import gradient_penalty, load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator, initialize_weights
from tqdm import tqdm
from dataset import MyImageDataset
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True


def train_fn(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    l1,
    vgg_loss,
    g_scaler,
    d_scaler,
    writer,
    tb_step,
):
    loop = tqdm(loader, leave=True)

    for idx, (lr, hr) in enumerate(loop):
        hr = hr.to(config.DEVICE)
        lr = lr.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            fake = gen(lr)
            disc_real = disc(hr)
            disc_fake = disc(fake.detach())
            gp = gradient_penalty(disc, hr, fake, device=config.DEVICE)  # ?? 相对loss
            loss_disc = config.LAMBDA_GP*gp - (torch.mean(disc_real) - torch.mean(disc_fake))

        opt_disc.zero_grad()
        d_scaler.scale(loss_disc).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()


        # Train Generator
        with torch.cuda.amp.autocast():
            l1_loss = 1e-2 * l1(fake, hr)
            adversarial_loss = 5e-3 * -torch.mean(disc(fake))
            vgg_for_loss = vgg_loss(fake, hr)
            loss_gen = l1_loss + adversarial_loss + vgg_for_loss

        opt_gen.zero_grad()
        g_scaler.scale(loss_gen).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        writer.add_scalar("Disc loss", loss_disc.item(), global_step=tb_step)
        tb_step += 1

        if idx % 1 == 0:
            plot_examples("data/val", gen)

        loop.set_postfix(
            gp=gp.item(),
            disc=loss_disc.item(),
            l1=l1_loss.item(),
            vgg=vgg_for_loss.item(),
            adversarial=adversarial_loss.item(),
        )

    return tb_step


def main():
    dataset = MyImageDataset(root_img_dir="data/hr")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    initialize_weights(gen)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    writer = SummaryWriter("logs")
    tb_step = 0
    l1 = nn.L1Loss()
    gen.train()
    disc.train()
    vgg_loss = VGGLoss()

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):
        tb_step = train_fn(
            loader,
            disc,
            gen,
            opt_gen,
            opt_disc,
            l1,
            vgg_loss,
            g_scaler,
            d_scaler,
            writer,
            tb_step,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    try_model = True

    if try_model:
        gen = Generator(in_channels=3).to(config.DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        gen.eval()
        plot_examples("data/val", gen)
    else:
        main()