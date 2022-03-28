import torch

class Config():
    epoch = 5
    batch_size = 512
    lr = 1e-4

    trainPath = '../data/train.mat'
    valPath = '../data/val.mat'
    netPath = './resnet.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")