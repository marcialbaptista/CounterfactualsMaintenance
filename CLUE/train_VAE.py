#%% import dependencies
import glob
import sys
import os
import csv
import json

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd

from torch import nn, save, load
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from bayesian_torch.models.dnn_to_bnn import get_kl_loss
import bayesian_torch.layers as bl

from CLUE_master.VAE.fc_gauss import VAE_gauss_net
from CLUE_master.VAE.train import train_VAE
from CLUE_master.src.utils import Datafeed


from variables import *

torch.manual_seed(42)
current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory

TRAINDATASET = f'data/{DATASET}/min-max/train'
TESTDATASET = f'data/{DATASET}/min-max/test'

save_dir = os.path.join(project_path, f'/CLUE/VAE_model_states/{DATASET}/VAE_model_state_test')
save_dir = 'CLUE/saves/VAE_model_test'

#%% main script
if __name__ == '__main__':

    print('Processing training and testing data')
    x_train = []
    x_test = []

    y_train = []
    y_test = []

    for testtrain in [TRAINDATASET, TESTDATASET]:
        #Import input file paths
        file_paths = glob.glob(os.path.join(project_path, testtrain, '*.txt'))  # Get a list of all file paths in the folder
        file_paths.sort()
        file_paths = file_paths[0:178]

        for file_path in file_paths:

            #load sample with true RUL
            sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
            sample_id = int(file_path[-13:-8])
            label = int(file_path[-7:-4])

            #Create labels for sensors and RUL
            sensors = [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
            head = [[f'Sensor {i,j}' for j in range(len(sample)) for i in sensors]]
            head[0].append('RUL')

            #Flatten sample and combine with RUL
            sample = [[element for row in sample for element in row]] #flatten time series sample into format [(sensor 1, timestep 0),...(sensor n, timestep w)]
            # sample = np.column_stack((sample, label))

            if testtrain == TRAINDATASET:
                x_train.append(sample)
                y_train.append(label)
            elif testtrain == TESTDATASET:
                x_test.append(sample)
                y_test.append(label)

    x_train = np.array(x_train)
    x_train = x_train[:,0,:].astype(np.float32) #2D array of flattend training inputs

    x_test = np.array(x_test)
    x_test = x_test[:,0,:].astype(np.float32)

    y_train = np.array(y_train)
    # y_train = y_train[:,0,:].astype(np.float32) #2D array of flattend training inputs

    y_test = np.array(y_test)
    # y_test = y_test[:,0,:].astype(np.float32)

    trainset = Datafeed(x_train, y_train, transform=None)
    valset = Datafeed(x_test, y_test, transform=None)

    input_dim = x_train.shape[1]

    batch_size = 128
    nb_epochs = 2500
    early_stop = 200
    lr = 1e-4

    cuda = torch.cuda.is_available()

    net = VAE_gauss_net(input_dim=input_dim, width=300, depth=3, latent_dim=2, pred_sig=False, lr=lr, cuda=cuda)

    vlb_train, vlb_dev = train_VAE(net, save_dir, batch_size, nb_epochs, trainset, valset,
                                       cuda=cuda, flat_ims=False, train_plot=False, early_stop=early_stop)