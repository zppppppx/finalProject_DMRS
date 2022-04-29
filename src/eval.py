import os
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DMRSdata
import Config
from model.DCNN import DCNN
from model.ResNet import ResNet
from model.DenseNet import DenseNet
from model.Autoencoder import Autoencoder
from model.Transformer import Transformer

opt = Config.Config()
dcnn = DCNN()
resnet = ResNet()
densenet = DenseNet()
autoencoder = Autoencoder()
transformer = Transformer(192, 192, n_layers=1)
evalNet = transformer
evalNet = evalNet.to(opt.device)

if os.path.exists(opt.valNet):
    state = torch.load(opt.valNet)
    evalNet.load_state_dict(state)

dmrsData = DMRSdata.DMRS(opt.valPath, 'picture_l')
dmrsLoader = DataLoader(dataset=dmrsData, batch_size=opt.batch_size, shuffle=True)

criterion = nn.MSELoss(reduction='sum').to(opt.device)

with torch.no_grad():
    runningLoss = 0.
    runningSum = 0.

    for idx, data in enumerate(dmrsLoader, 0):
        H_in, H_out, snr = data
        H_in = H_in.to(opt.device)
        # H_in = util.Upsample(H_in, (14, 192))
        H_out = H_out.to(opt.device)
        
        start = time.time()
        H_out_calc = evalNet(H_in, snr)
        end = time.time()
        interval = end - start

        

        loss = criterion(H_out, H_out_calc)
        square = H_in*H_in
        dims = list(range(square.dim()))
        runningSum += square.sum(dims)

        runningLoss += loss.item()

        if idx %10 == 9:
            print('Idx: %d, running loss: %.4f, running time: %.5f' % (idx, runningLoss/runningSum, interval))

