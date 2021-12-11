import torch
import h5py
from torch.utils.data import Dataset

class DMRS(Dataset):
    def __init__(self, filePath):
        with h5py.File(filePath,'r') as f:
            self.H_in = torch.tensor(f['H_in'], dtype=torch.float32).permute([3,0,1,2])
            self.H_out = torch.tensor(f['H_out'], dtype=torch.float32).permute([3,0,1,2])
            self.N = f['N'][0]
            self.SNR = f['SNR'][0]
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.H_in[index], self.H_out[index], self.SNR[index]