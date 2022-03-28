import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class GELU(nn.Module):
    """
    Activation Function.
    """
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(torch.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

class decoder_block(nn.Module):
    def __init__(self, Up_size, in_channel, out_channel):
        super(decoder_block, self).__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(Up_size, mode='bilinear'),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.decoder(x)
        return out


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 32, 3),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,3)) ## 64*2*15
        )
        self.encoder_fl = nn.Sequential(
            nn.Linear(64*2*15, 64*2*15),
            nn.ReLU(inplace=True),
            nn.Linear(64*2*15, 64*2*15),
            nn.ReLU(inplace=True),
            nn.Linear(64*2*15, 64*2*15)
        )


        self.decoder = nn.Sequential(
            decoder_block((6, 47), 64, 32),
            decoder_block((14, 96), 32, 2)
        )
        
    def forward(self, x):
        up_size = (14, 96)
        x = nn.Upsample(size=up_size, mode='bilinear')(x)

        # Encoder
        x = self.encoder_conv(x)
        x = x.view(-1, 64*2*15)
        x = self.encoder_fl(x)

        # Decoder
        x = x.view(-1, 64, 2, 15)
        x = self.decoder(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2*2*48, 2*2*24),
            nn.GELU(),
            nn.Linear(2*2*24, 2*2*12),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2*2*12, 2*2*24),
            nn.GELU(),
            nn.Linear(2*2*24, 2*2*48),
            nn.GELU(),
            nn.Linear(2*2*48, 2*2*96),
            nn.GELU(),
            nn.Linear(2*2*96, 2*14*96)
        )

    def forward(self, x):
        x = x.view(-1, 2*2*48)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 2, 14, 96)
        return x

if __name__ == '__main__':
    x = torch.ones(32,2,2,48)
    net = autoencoder()
    out = net(x)
    print(out.shape)