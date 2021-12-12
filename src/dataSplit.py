import numpy as np
import h5py
import scipy.io as io

filePath = '../data/Training_Data.mat'
trainPath = '../data/train.mat'
valPath = '../data/val.mat'
ratio = 0.9 # split ratio

with h5py.File(filePath,'r') as f:
    H_in = np.array(f['H_in'])
    H_out = np.array(f['H_out'])
    number = np.array(f['N'])
    SNR = np.array(f['SNR'])

    permutation = np.random.permutation(H_in.shape[-1])
    H_in_shuffle = H_in[:, :, :, permutation]
    H_out_shuffle = H_out[:, :, :, permutation]
    SNR_shuffle = SNR[:, permutation]

    trainLength = int(number[0][0]*ratio)
    trainData = {'H_in':H_in[:,:,:,:trainLength],'H_out':H_out[:,:,:,:trainLength],
                'N':np.array([trainLength,],dtype=np.float), 'SNR':SNR[:,:trainLength]}
    with h5py.File(trainPath, 'w') as g:
        for k, v in trainData.items():
            g.create_dataset(name=k, data=v)
    # io.savemat(trainPath, trainData)

    valData = {'H_in':H_in[:,:,:,trainLength:],'H_out':H_out[:,:,:,trainLength:],
                'N':np.array([number[0][0]-trainLength,],dtype=np.float), 'SNR':SNR[:,trainLength:]}
    with h5py.File(valPath, 'w') as g:
        for k, v in valData.items():
            g.create_dataset(name=k, data=v)
    # io.savemat(valPath, valData)


