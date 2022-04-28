from random import randrange
from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_ch, growth_rate, numlayers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[DenseLayer(in_ch+growth_rate*i, growth_rate) for i in range(numlayers)]
        )
        self.lff = nn.Conv2d(in_ch+growth_rate*numlayers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x+self.lff(self.layers(x))

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.feature = 16
        self.growth_rate = 16
        self.numblocks = 4
        self.numlayers = 4

        # shalow feature extraction
        self.sfe1 = nn.Conv2d(2, self.feature, 3, 1, 1)
        self.sfe2 = nn.Conv2d(self.feature, self.feature, 3, 1, 1)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.feature, self.growth_rate, self.numlayers)])
        for _ in range(self.numblocks-1):
            self.rdbs.append(RDB(self.growth_rate, self.growth_rate, self.numlayers))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.growth_rate*self.numblocks, self.feature, 1),
            nn.Conv2d(self.feature, self.feature, 3, 1, 1)
        )

        self.out = nn.Conv2d(self.feature, 2, 3, 1, 1)

    def forward(self, x, snr):
        x = nn.UpsamplingNearest2d(size=(14, 96))(x)
        sfe1 = self.sfe1(x)
        x = self.sfe2(sfe1)

        local_features = []
        for i in range(self.numblocks):
            x = self.rdbs[i](x)
            local_features.append(x)
        x = self.gff(torch.cat(local_features, 1)) + sfe1
        x = self.out(x)
        return x

        



