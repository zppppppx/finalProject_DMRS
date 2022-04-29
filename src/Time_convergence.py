from pickle import FALSE
import torch
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
trainData = DMRSdata.DMRS(opt.trainPath, 'picture_l')
valData = DMRSdata.DMRS(opt.valPath, 'picture_l')
# h_in, _, _ = dmrsData[10]
# print(h_in.shape)
trainLoader = DataLoader(dataset=trainData, batch_size=opt.batch_size, shuffle=True)
valLoader = DataLoader(dataset=valData, batch_size=opt.batch_size, shuffle=True)

transformer = Transformer.Transformer(192, 192, n_layers=1).to(opt.device)
if os.path.exists(opt.netPath):
    state = torch.load(opt.netPath)
    transformer.load_state_dict(state)

optimizer = optim.Adam(transformer.parameters(), lr=opt.lr)
criterion = nn.MSELoss(reduction='sum').to(opt.device)

train = True
if train:
    trainingLoss = torch.tensor([])
    valLoss = torch.tensor([])
    for epoch in range(opt.epoch):
        runningLoss = 0.
        runningSum = 0.

        for idx, data in enumerate(trainLoader, 0):
            H_in, H_out, snr = data
            H_in = H_in.to(opt.device)
            H_out = H_out.to(opt.device)

            H_out_calc = transformer(H_in, snr)

            optimizer.zero_grad()
            loss = criterion(H_out, H_out_calc)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()#*opt.batch_size
            square = H_in*H_in
            dims = list(range(square.dim()))
            runningSum += square.sum(dims)

            if idx %10 == 9:
                nmse = runningLoss/runningSum
                print('Epoch: %d, idx: %d, running loss: %.7f' % (epoch, idx, nmse))
                

            state = transformer.state_dict()
            torch.save(state, opt.netPath)

        nmse = (runningLoss/runningSum).clone().detach().cpu()
        nmse = torch.tensor([nmse])
        trainingLoss = torch.cat((trainingLoss, nmse), dim=0)
        print('Epoch %d: training loss :'%(epoch), trainingLoss)

        with torch.no_grad():
            runningLoss = 0.
            runningSum = 0.

            for idx, data in enumerate(valLoader, 0):
                H_in, H_out, snr = data
                H_in = H_in.to(opt.device)
                H_out = H_out.to(opt.device)
                
                H_out_calc = transformer(H_in, snr)

                loss = criterion(H_out, H_out_calc)
                square = H_in*H_in
                dims = list(range(square.dim()))
                runningSum += square.sum(dims)

                runningLoss += loss.item()#*opt.batch_size
                
                if idx %10 == 9:
                    print('Idx: %d, running loss: %.7f' % (idx, runningLoss/runningSum))

            nmse = (runningLoss/runningSum).clone().detach().cpu()
            nmse = torch.tensor([nmse])
            valLoss = torch.cat((valLoss, nmse), dim=0)
            print('Epoch %d: eval loss :'%(epoch), valLoss)
        
    torch.save([trainingLoss, valLoss], './result/lossTime.pt')