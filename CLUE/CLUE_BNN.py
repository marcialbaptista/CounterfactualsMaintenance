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

from CLUE_master.interpret.CLUE import CLUE
from CLUE_master.interpret.visualization_tools import latent_map_2d_gauss, latent_project_gauss, latent_project_cat
from CLUE_master.src.utils import Ln_distance
from CLUE_master.src.probability import decompose_std_gauss, decompose_entropy_cat

from CLUE_master.BNN.models import CustomBayesianNeuralNetwork
from CLUE_master.BNN.wrapper import BNN_gauss, BNN_cat, MLP


from variables import *

torch.manual_seed(42)
current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory

TRAINDATASET = f'data/{DATASET}/min-max/train'
TESTDATASET = f'data/{DATASET}/min-max/test'

# save_dir = os.path.join(project_path, f'/CLUE/VAE_model_states/{DATASET}/VAE_model_state_test')


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
        file_paths = file_paths[0:178] #TEMP: select first engine

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
                y_train.append([label])
            elif testtrain == TESTDATASET:
                x_test.append(sample)
                y_test.append([label])

    x_train = np.array(x_train)
    x_train = x_train[:,0,:].astype(np.float32) #2D array of flattend training inputs

    x_test = np.array(x_test)
    x_test = x_test[:,0,:].astype(np.float32) #2D array of flattend testing inputs

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    

    trainset = Datafeed(x_train, y_train, transform=None)
    valset = Datafeed(x_test, y_test, transform=None)


    cuda = torch.cuda.is_available()

    #Load trained VAE
    save_dir = 'CLUE/saves/VAE_model_test_models'
    lr = 1e-4
    input_dim = x_train.shape[1]
    VAE = VAE_gauss_net(input_dim=input_dim, width=300, depth=3, latent_dim=2, pred_sig=False, lr=lr, cuda=cuda)

    VAE.load(project_path + '/' + save_dir + "/theta_best.dat")

    #Load trained BNN
    # Model input parameters
    input_size = 14 #number of features
    hidden_size = 32

    model = CustomBayesianNeuralNetwork(input_size=input_size, hidden_size=hidden_size)
    N_train = x_train.shape[0]
    lr = 1e-2
    BNN = BNN_gauss(model, N_train, lr, cuda)

    BNN.load_weights(f'{project_path}/BNN/model_states/BNN_model_state_{DATASET}_test.pkl')

    #%% run CLUE explainer
    print('Calculating uncertanties')
    tr_aleatoric_vec, tr_epistemic_vec, z_train, x_train, y_train = \
        latent_project_gauss(BNN, VAE, dset=trainset, batch_size=2048, cuda=cuda, prob_BNN=True)
    
    tr_uncertainty_vec = tr_aleatoric_vec + tr_epistemic_vec

    te_aleatoric_vec, te_epistemic_vec, z_test, x_test, y_test = \
        latent_project_gauss(BNN, VAE, dset=valset, batch_size=2048, cuda=cuda, prob_BNN=True)
    
    te_uncertainty_vec = (te_aleatoric_vec**2 + te_epistemic_vec**2)**(1.0/2)

    uncertainty_idxs_sorted = np.flipud(np.argsort(te_uncertainty_vec))
    aleatoric_idxs_sorted = np.flipud(np.argsort(te_aleatoric_vec))
    epistemic_idxs_sorted = np.flipud(np.argsort(te_epistemic_vec))


    torch.cuda.empty_cache()
    
    use_index = uncertainty_idxs_sorted 

    Nbatch = 512
    z_init_batch = z_test[use_index[:Nbatch]]
    x_init_batch = x_test[use_index[:Nbatch]]
    y_init_batch = y_test[use_index[:Nbatch]]


    dist = Ln_distance(n=1, dim=(1))
    x_dim = x_init_batch.reshape(x_init_batch.shape[0], -1).shape[1]


    aleatoric_weight = 0
    epistemic_weight = 0
    uncertainty_weight = 1

    distance_weight = 2 / x_dim
    prediction_similarity_weight = 0

    print('Running CLUE')
    CLUE_explainer = CLUE(VAE, BNN, x_init_batch, uncertainty_weight=uncertainty_weight, aleatoric_weight=aleatoric_weight, epistemic_weight=epistemic_weight,
                      prior_weight=0, distance_weight=distance_weight,
                 latent_L2_weight=0, prediction_similarity_weight=prediction_similarity_weight,
                 lr=1e-2, desired_preds=y_init_batch, cond_mask=None, distance_metric=dist,
                 z_init=None, norm_MNIST=False,
                 flatten_BNN=False, regression=True, cuda=cuda)
    
    torch.autograd.set_detect_anomaly(False)

    z_vec, x_vec, uncertainty_vec, epistemic_vec, aleatoric_vec, cost_vec, dist_vec = CLUE_explainer.optimise(
                                                        min_steps=3, max_steps=100,
                                                        n_early_stop=3)
    
    print(x_vec)