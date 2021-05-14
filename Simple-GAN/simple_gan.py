import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),  # 输入in_features 748， 748 -> 128
            nn.LeakyReLU(0.01),  # 
            nn.Linear(128, 1), # 128 -> 1
            nn.Sigmoid(),   # 将实数映射到[0,1]区间
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, image_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),   # z_dim维 升至 256维
            nn.LeakyReLU(0.01),
            nn.Linear(256, image_dim), # 256维 升至 image_dim维度
            nn.Tanh(),  # Tanh使得生成数据范围在[-1, 1]，因为真实数据经过transforms后也是在这个区间
        )

    def forward(self, x):
        return self.gen(x)


# 超参数
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 100

Disc = Discriminator(image_dim).to(device)
Gen = Generator(z_dim, image_dim).to(device)
opt_disc = optim.Adam(Disc.parameters(), lr=lr)
opt_gen = optim.Adam(Gen.parameters(), lr=lr)
criterion = nn.BCELoss()  # 单目标二分类交叉熵函数

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

fixed_noise = torch.randn((batch_size, z_dim)).to(device)
write_fake = SummaryWriter(f'logs/fake')
write_real = SummaryWriter(f'logs/real')
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]
        ## D: 目标：真的判断为真，假的判断为假
        ## 训练Discriminator: max log(D(x)) + log(1-D(G(z)))
        disc_real = Disc(real).view(-1)  # 将真实图片放入到判别器中
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))  # 真的判断为真

        noise = torch.randn(batch_size, z_dim).to(device)   
        fake = Gen(noise)  # 将随机噪声放入到生成器中
        disc_fake = Disc(fake).view(-1)  # 识别器判断真假
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))  # 假的应该判断为假
        lossD = (lossD_real + lossD_fake) / 2  # loss包括判真损失和判假损失

        Disc.zero_grad()   # 在反向传播前，先将梯度归0
        lossD.backward(retain_graph=True)  # 将误差反向传播
        opt_disc.step()   # 更新参数

        # G： 目标：生成的越真越好
        ## 训练生成器： min log(1-D(G(z))) <-> max log(D(G(z)))
        output = Disc(fake).view(-1)   # 生成的放入识别器
        lossG = criterion(output, torch.ones_like(output))  # 与“真的”的距离，越小越好
        Gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        # 输出一些信息，便于观察
        if batch_idx == 0:

            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)}' \
                    Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = Gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                write_fake.add_image(
                    "Mnist Fake Image", img_grid_fake, global_step=step
                )
                write_real.add_image(
                    "Mnist Real Image", img_grid_real, global_step=step
                )
                step += 1


