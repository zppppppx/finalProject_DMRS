import torch
import torch.nn as nn

def Upsample(inputs, size):
    return nn.UpsamplingNearest2d(size)(inputs)