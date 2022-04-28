from pickle import FALSE
import torch
import numpy
import util
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DMRSdata
import Config
import torch.optim as optim
from model import Transformer
import os

opt = Config.Config()
dmrsData = DMRSdata.DMRS(opt.trainPath, 'picture_l')
# h_in, _, _ = dmrsData[10]
# print(h_in.shape)
dmrsLoader = DataLoader(dataset=dmrsData, batch_size=opt.batch_size, shuffle=True)

transformer = Transformer.Transformer(192, 192, n_layers=1).to(opt.device)
if os.path.exists(opt.netPath):
    state = torch.load(opt.netPath)
    transformer.load_state_dict(state)

optimizer = optim.Adam(transformer.parameters(), lr=opt.lr)
criterion = nn.MSELoss().to(opt.device)

train = True
if train:
    for epoch in range(opt.epoch):
        runningLoss = 0.

        for idx, data in enumerate(dmrsLoader, 0):
            H_in, H_out, snr = data
            H_in = H_in.to(opt.device)
            # H_in = util.Upsample(H_in, (14, 192))
            H_out = H_out.to(opt.device)

            H_out_calc = transformer(H_in, snr)

            optimizer.zero_grad()
            loss = criterion(H_out, H_out_calc)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()

            if idx %10 == 9:
                print('Epoch: %d, idx: %d, running loss: %.4f' % (epoch, idx, runningLoss))
                runningLoss = 0.

            state = transformer.state_dict()
            torch.save(state, opt.netPath)