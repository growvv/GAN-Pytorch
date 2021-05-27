import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),

            CNNBlock(64, 128, 2),
            CNNBlock(128, 256, 2),
            CNNBlock(256, 512, 1),

            nn.Conv2d(512, 1, 4, 1, 1, padding_mode="reflect"),
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.model(x)  
        return x  # 输出的30*30的patch，而不是1*1


def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    disc = Discriminator()
    preds = disc(x, y)
    print(disc)
    print(preds.shape)

if __name__ == "__main__":
    test()
