from pickle import FALSE
import torch
import numpy
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DMRSdata
import Config
import torch.optim as optim
from model import Autoencoder
import os


opt = Config.Config()
dmrsData = DMRSdata.DMRS(opt.trainPath)
# h_in, _, _ = dmrsData[10]
# print(h_in.shape)
dmrsLoader = DataLoader(dataset=dmrsData, batch_size=opt.batch_size, shuffle=True)

autoencoder = Autoencoder.Autoencoder().to(opt.device)
if os.path.exists(opt.netPath):
    state = torch.load(opt.netPath)
    autoencoder.load_state_dict(state)

optimizer = optim.Adam(autoencoder.parameters(), lr=opt.lr)
criterion = nn.MSELoss().to(opt.device)

train = False
if train:
    for epoch in range(opt.epoch):
        runningLoss = 0.

        for idx, data in enumerate(dmrsLoader, 0):
            H_in, H_out, snr = data
            H_in = H_in.to(opt.device)
            H_out = H_out.to(opt.device)

            H_out_calc = autoencoder(H_in)

            optimizer.zero_grad()
            loss = criterion(H_out, H_out_calc)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()

            if idx %10 == 9:
                print('Epoch: %d, idx: %d, running loss: %.4f' % (epoch, idx, runningLoss))
                runningLoss = 0.

            state = autoencoder.state_dict()
            torch.save(state, opt.netPath)


h_in, h_out, _ = dmrsData[9]
h_in = h_in[None,].to(opt.device)
h_out_calc = autoencoder(h_in)
plt.figure(figsize=(8, 3))
plt.subplot(1,2,1)
# plt.subplot('position', [0.05,0.05,0.45,0.95])
plt.imshow(h_out[0].cpu(), aspect=48/7)
plt.subplot(1,2,2)
# plt.subplot('position', [0.55,0.05,0.95,0.95])
plt.imshow(h_out_calc[0][0].cpu().detach(), aspect=48/7)
plt.show()

# fig, axes = plt.subplots(12)
# plt.imshow(h_out[0].cpu(), ax)
