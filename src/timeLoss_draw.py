import matplotlib.pyplot as plt
import torch
import numpy as np

trainingLoss, valLoss = torch.load('./result/lossTime.pt')
trainingLoss = trainingLoss.clone().detach().numpy()
valLoss = valLoss.clone().detach().numpy()


plt.plot(trainingLoss)
plt.plot(valLoss)
plt.title('Train History')
plt.ylabel('Loss NMSE')
plt.xlabel('Epoch')
plt.legend(['train','validation'],loc = 'upper right')
plt.show()

