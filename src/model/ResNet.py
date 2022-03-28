from random import randrange
from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn

class ResBlock(nn.Module):
    """
    Basic block for ResNet.
    """
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)

    def forward(self, x):
        origin = x
        out = self.conv2(self.relu(self.conv1(x)))
        out = torch.add(origin, out*0.1)

        return out

class ResNet(nn.Module):
    def __init__(self, resNum=8):
        super(ResNet, self).__init__()
        self.conv_input = nn.Conv2d(2, 32, 3, padding=1, bias=False)
        self.res = self._make_layer(ResBlock, resNum)
        self.conv_mid = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.conv_output = nn.Conv2d(32, 2, 3, 1, 1, bias=False)

    def forward(self, x, snr):
        out = nn.UpsamplingBilinear2d(size=(14,96))(x)
        out = self.conv_input(out)
        res = out
        out = self.conv_mid(self.res(out))
        out = torch.add(out, res)
        out = self.conv_output(out)
        return out

    def _make_layer(self, block, numlayers):
        layers = []
        for _ in range(numlayers):
            layers.append(block())
        return nn.Sequential(*layers)
