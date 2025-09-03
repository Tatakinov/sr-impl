#!/bin/python3

import numpy as np
import os
from PIL import Image, ImageFilter
import sys
import torch
from torchvision import transforms
from torchvision.utils import save_image

from net import Net

net = Net()

net.load_state_dict(torch.load(sys.argv[1]))

net.eval()

image = Image.open('input.png')
image = image.convert("RGBA")

to_tensor = transforms.ToTensor()

r = to_tensor(image.getchannel("R")).unsqueeze(0)
g = to_tensor(image.getchannel("G")).unsqueeze(0)
b = to_tensor(image.getchannel("B")).unsqueeze(0)
a = to_tensor(image.getchannel("A")).unsqueeze(0)

rx2 = net(r).squeeze()
rx2 = rx2.mul(255).add_(0.5).clamp_(0, 255).byte()
rx2 = Image.fromarray(rx2.numpy(), mode = "L")

gx2 = net(g).squeeze()
gx2 = gx2.mul(255).add_(0.5).clamp_(0, 255).byte()
gx2 = Image.fromarray(gx2.numpy(), mode = "L")

bx2 = net(b).squeeze()
bx2 = bx2.mul(255).add_(0.5).clamp_(0, 255).byte()
bx2 = Image.fromarray(bx2.numpy(), mode = "L")

ax2 = net(a).squeeze()
ax2 = ax2.mul(255).add_(0.5).clamp_(0, 255).byte()
ax2 = Image.fromarray(ax2.numpy(), mode = "L")

imagex2 = Image.merge("RGBA", [rx2, gx2, bx2, ax2])

imagex2.save("output.png")
