from scipy.io import loadmat
import torch
import h5py
from torch import tensor
import torch.nn as nn


filePath = '../data/Training_Data.mat'

a = torch.rand([2,2,3])
b = a*a
print(b.shape)
dims = list(range(b.dim()))
print(b.sum(dims))

a = torch.tensor([2])
b = torch.tensor(1)
# b = torch.tensor([b])

# torch.save([a, b], './test.pt')
c, d = torch.load('./test.pt')
print(c, d)

# with h5py.File(filePath,'r') as f:
#     print(f.keys())
#     print(f['H_out'].shape)
#     H_in = f['H_in'][:,:,:,:1]
#     H_out = f['H_out'][:,:,:,:1]
#     H_in = torch.tensor(H_in, dtype=torch.float32)
#     H_out = torch.tensor(H_out, dtype=torch.float32)
#     # print(H_in[:,:,:2,:],'\n',H_out[:,[3,11],:2:4,:])

# # # print(torch.pi)
# a = [1, *(2 for i in range(10))]
# print(a)

# conv1 = nn.ConvTranspose2d(1,1,1,(7,1),(3,3))
# inputs = torch.rand((32,1,2,48))
# output = conv1(inputs)
# print(output.shape)

# # a = torch.rand((1,3,2))
# # print(a.shape)
# # print(a.T.shape)

# # a = torch.rand((32, 2, 96))
# # out = nn.UpsamplingNearest2d(size=(14, 192))(a[:,None,:]).squeeze()
# # print(out.shape)
# # a = [[2, 32], [32, 64], *([64, 64] for _ in range(9-4)), [64, 32], [32, 2]]
# # print(a)