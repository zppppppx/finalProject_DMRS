from scipy.io import loadmat
import torch
import h5py
from torch import tensor
import torch.nn as nn

filePath = '../data/Training_Data.mat'


with h5py.File(filePath,'r') as f:
    print(f.keys())
    print(f['H_out'].shape)
    H_in = f['H_in'][:,:,:,:1]
    H_out = f['H_out'][:,:,:,:1]
    H_in = torch.tensor(H_in, dtype=torch.float32)
    H_out = torch.tensor(H_out, dtype=torch.float32)
    # print(H_in[:,:,:2,:],'\n',H_out[:,[3,11],:2:4,:])

# # print(torch.pi)
a = [1, *(2 for i in range(10))]
print(a)

# a = torch.rand((1,3,2))
# print(a.shape)
# print(a.T.shape)

# a = torch.rand((32, 2, 2, 48))
# out = nn.UpsamplingNearest2d(size=(14, 96))(a)
# print(out.shape)
# a = [[2, 32], [32, 64], *([64, 64] for _ in range(9-4)), [64, 32], [32, 2]]
# print(a)