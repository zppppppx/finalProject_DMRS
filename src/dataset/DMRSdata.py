import torch
import h5py
from torch.utils.data import Dataset
import numpy as np

class DMRS(Dataset):
    """
    Build the dataset.

    Args:
        filePath: the path of *.mat.

    Return:
        self.H_in: DMRS sources, whose dimension is [batch, real/imag, time, freq] or [batch, channel(1), time, real/imag-freq]
        self.H_out: estimated channel information, whose dimension is [batch, real/imag, time, freq] or [batch, channel(1), time, real/imag-freq]
    """
    def __init__(self, filePath, mode='picture_m'):
        with h5py.File(filePath,'r') as f:
            if mode=='picture_m':
                self.H_in = torch.tensor(f['H_in'], dtype=torch.float32).permute([3,0,1,2])
                self.H_out = torch.tensor(f['H_out'], dtype=torch.float32).permute([3,0,1,2])
            if mode=='picture_l':
                self.H_in = torch.tensor(f['H_in'], dtype=torch.float32).permute([3,1,0,2]).reshape((-1, 1, 2, 96))
                self.H_out = torch.tensor(f['H_out'], dtype=torch.float32).permute([3,1,0,2]).reshape((-1, 1, 14, 192))
            self.N = f['N'][0].astype(np.int)
            self.SNR = f['SNR'][0][:,None]
            # E_real, E_imag = self._expectation(self.H_in[:,0,:,:]), self._expectation(self.H_in[:,1,:,:])
            
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.H_in[index], self.H_out[index], self.SNR[index]
    
    def _expectation(self, x):
        for i in range(len(x.shape)):
            x = x.sum(-1)
        return x

    
if __name__ == '__main__':
    trainPath = '../data/train.mat'
    trainData = DMRS(trainPath)
    H_in, H_out, SNR = trainData[:3]
    print(H_in)