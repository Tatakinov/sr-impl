#!/usr/bin/python3

import numpy as np
import os
from PIL import Image, ImageFilter
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random

from net import Net

max_epoch = 400
batch_size = 64 # mini batch size

class GetDataset():
    def __init__(self, mode, crop, resize):
        self.crop = crop
        self.resize = resize
        self.to_tensor = transforms.ToTensor()
        self.files = [os.path.join(mode, item) for item in os.listdir(mode) if item.endswith(".jpg") or item.endswith(".jpeg") or item.endswith(".png")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = Image.open(self.files[index])
        image = self.crop(item)
        reduce_image = self.resize(image)
        index = random.randint(0, 2)
        image = image.getchannel(index)
        reduce_image = reduce_image.getchannel(index)
        return [self.to_tensor(reduce_image), self.to_tensor(image)]

def main():
    reduce_size = 64
    size = 128

# new
    net = Net()

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        print("load model")
        net.load_state_dict(torch.load(sys.argv[1]))

# data loader
    train = GetDataset("train", transforms.Compose([
        transforms.RandomCrop(size = (size, size)),
        transforms.RandomHorizontalFlip(),
        lambda x: x.rotate(random.randint(0, 3) * 90),
        ]), transforms.Compose([
        transforms.Resize(reduce_size),
        ]))
    train_loader = DataLoader(train, batch_size, shuffle=True)
    test = GetDataset("test", transforms.Compose([
        transforms.RandomCrop(size = (size, size)),
        ]), transforms.Compose([
        transforms.Resize(reduce_size),
        ]))
    test_loader = DataLoader(test, batch_size, shuffle=False)

    net.train()

# Loss function
    criterion = nn.L1Loss()

# Stochastic gradient descent
    optimizer = torch.optim.Adam(net.parameters())

    #"""

# Train
    print("+++TRAIN+++")
    for epoch in range(max_epoch+1) :
        for batch in train_loader:
            x, t = batch
            #clear grad
            optimizer.zero_grad() 
            #forward
            y = net(x)
            #loss function
            loss = criterion(y, t)
            #BP
            loss.backward()
            #update
            optimizer.step()
        print("epoc:", epoch, ' loss:', loss.item())

        if epoch % 10 == 0:
            print("---TEST---")
            net.eval()
            loss_list = []
            for batch in test_loader:
                x, t = batch
                #forward
                y = net(x)
                #loss function
                loss = criterion(y, t)
                loss_list.append(loss.item())
            print("loss:", sum(loss_list) / len(loss_list))
            net.train()

        # save
        torch.save(net.state_dict(), 'model' + str(epoch) + '.bin')

main()
