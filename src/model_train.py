import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from model_design import *

'''=============== You need to configure here: ====================================='''
# Set the data set path and channel configurations
# 'data_path' need to match the saved path of the downloaded data set
data_path = ''

# Load data.
# DMRS_half.npy: received signal on DMRS position with half DMRS density in frequency
DMRS_pattern = np.load(os.path.join(data_path, 'DMRS_half.npy'))
# DMRS_qtr.npy: received signal on DMRS position with quarter DMRS density in frequency
# DMRS_pattern = np.load(os.path.join(data_path, 'DMRS_qtr.npy'))

# H.npy: ideal channel on PDSCH.
H_true = np.load(os.path.join(data_path, 'H.npy'))
# Here we use all data set to train by default.
# But, you need to divide the training set and test set.

'''================================================================================'''
# Build model
CE_input = keras.Input(shape=DMRS_pattern.shape[1:])
CE_output = CE_net(CE_input)
CE_NET = keras.Model(inputs=CE_input, outputs=CE_output, name='CE_NET')
CE_NET.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')
CE_NET.summary()

# Model training
CE_NET.fit(x=DMRS_pattern, y=H_true, batch_size=512, epochs=200, verbose=2, validation_split=0.01)
