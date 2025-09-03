import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, filter_size = 64, scale_factor = 1.0):
        super(ResidualBlock, self).__init__()
        self.scale_factor = scale_factor
        self.l1 = nn.Conv2d(filter_size, filter_size, kernel_size = 3, stride = 1, padding = 1)
        self.l2 = nn.Conv2d(filter_size, filter_size, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        x0 = x
        x = self.l1(x)
        x = nn.LeakyReLU()(x)
        x = self.l2(x)
        x = x * self.scale_factor + x0
        return x

class Net(nn.Module):
    def __init__(self, filter_size = 64, block_size = 8, scale_factor = 1.0):
        super(Net, self).__init__()
        self.input = nn.Conv2d(1, filter_size, kernel_size = 3, stride = 1, padding = 1)
        self.middle = nn.Conv2d(filter_size, filter_size, kernel_size = 3, stride = 1, padding = 1)
        self.block = nn.Sequential(*[ResidualBlock(filter_size = filter_size, scale_factor = scale_factor) for _ in range(block_size)])
        self.output = nn.Conv2d(filter_size, 2 ** 2, kernel_size = 1, stride = 1)
        self.post = nn.Conv2d(1, 1, kernel_size = 3, padding = 1, stride = 1)

    def forward(self, x):
        x = self.input(x)
        x0 = x
        for layer in self.block:
            x = layer(x)
        x = self.middle(x)
        x = x + x0
        x = self.output(x)
        x = nn.PixelShuffle(2)(x)
        x = self.post(x)
        return x
