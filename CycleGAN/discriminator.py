import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),  
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),

            Block(64, 128, stride=2),
            Block(128, 256, stride=2),
            Block(256, 512, stride=1),

            nn.Conv2d(512, 1, 4, 1, 1, padding_mode="reflect"),
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))


def test():
    x = torch.randn((5, 3, 256, 256))
    disc = Discriminator(in_channels=3)
    preds = disc(x)
    print(disc)
    print(preds.shape)


if __name__ == "__main__":
    test()
