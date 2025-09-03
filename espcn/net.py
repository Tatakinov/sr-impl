import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 5, padding = 2, stride = 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size = 3, padding = 1, stride = 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 2 * 2, kernel_size = 3, padding = 1, stride = 1),
            nn.PixelShuffle(2),
            )

    def forward(self, x):
        return self.layers(x)
