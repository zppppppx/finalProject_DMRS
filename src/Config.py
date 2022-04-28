import torch

class Config():
    epoch = 10
    batch_size = 256
    lr = 1e-4

    trainPath = '../data/train.mat'
    valPath = '../data/val.mat'
    netPath = './transformer_1.pth'
    valNet = './transformer_1.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")