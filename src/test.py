from scipy.io import loadmat
import torch
import h5py
from torch import tensor

filePath = '../data/Training_Data.mat'


with h5py.File(filePath,'r') as f:
    print(f.keys())
    print(f['SNR'].shape)
    H_in = f['H_in'][:,:,:,:1]
    H_out = f['H_out'][:,:,:,:1]
    H_in = torch.tensor(H_in, dtype=torch.float32)
    H_out = torch.tensor(H_out, dtype=torch.float32)
    # print(H_in[:,:,:2,:],'\n',H_out[:,[3,11],:2:4,:])

print(torch.pi)