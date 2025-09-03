#!/usr/bin/python3

import numpy as np
import os
from PIL import Image
import sys
import torch
import torch.nn as nn
import torch.onnx

from net import Net

def main():
# new
    net = Net()

    net.load_state_dict(torch.load(sys.argv[1]))

    net.eval()

    dummy_input = torch.randn(1, 1, 64, 64)

    with torch.no_grad():
        torch.onnx.export(net, dummy_input, 'model.onnx',
                          input_names = ['input'], output_names = ['output'],
                          dynamic_axes = {'input' : {0 : 'batch_size', 2 : 'width', 3 : 'height'},
                                          'output' : {0 : 'batch_size', 2 : 'width', 3 : 'height'}})

main()
