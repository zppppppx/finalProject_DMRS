import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(torch.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))
