from random import randrange
import torch
import torch.nn.functional as F
import torch.nn as nn

class DCNN(nn.Module):
    """
    Deep Convolutional Neural Network.
    """
    def __init__(self, numlayers=9):
        super(DCNN, self).__init__()
        self.numlayers = numlayers
        self.channels = [[2, 32], [32, 64], *([64, 64] for _ in range(self.numlayers-4)), [64, 32], [32, 2]]
        self.convLayers = nn.ModuleList([nn.Conv2d(self.channels[i][0], self.channels[i][1], kernel_size=3, padding=1) 
                                        for i in range(self.numlayers)])
        self.dcnn = nn.Sequential(nn.LeakyReLU(0.1, True))
        for layer in self.convLayers:
            self._initParam(layer)

    def forward(self, x, snr):
        x = nn.UpsamplingNearest2d(size=(14, 96))(x)
        for i in range(self.numlayers-1):
            x = self.convLayers[i](x)
            x = F.leaky_relu(x, 0.3)
        x = self.convLayers[-1](x)
        return x

    def _initParam(self, layer):
        nn.init.normal_(layer.weight, 0, 0.05)
        nn.init.constant_(layer.bias, 0)
