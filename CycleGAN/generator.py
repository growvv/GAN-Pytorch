import torch
import torch.nn as nn

class  ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)  if use_act else nn.Identity(),
        )
    
    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.residual = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        return x + self.residual(x)


class Generator(nn.Module):
    def __init__(self, img_channels=3, features=64, residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, features, 7, 1, 3, padding_mode="reflect"),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.Sequential(
            ConvBlock(features, features*2, kernel_size=3, stride=2, padding=1),
            ConvBlock(features*2, features*4, kernel_size= 3, stride=2, padding=1),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(features*4) for _ in range(residuals)]
        )

        self.up_blocks = nn.Sequential(
            ConvBlock(features*4, features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(features*2, features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        self.last = nn.Conv2d(features, img_channels, 7, 1, 3, padding_mode="reflect")


    def forward(self, x):
        x = self.initial(x)
        x = self.down_blocks(x)
        x = self.res_blocks(x)
        x = self.up_blocks(x)
        x = self.last(x)
        return torch.tanh(x)


def test():
    x = torch.randn((5, 3, 256, 256))
    gen = Generator(img_channels=3)
    preds = gen(x)
    print(gen)
    print(preds.shape)


if __name__ == "__main__":
    test()