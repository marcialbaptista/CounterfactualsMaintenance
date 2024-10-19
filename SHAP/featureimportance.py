#%% import dependencies
import sys
import os
import glob

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

import shap
import torch
import numpy as np

from BNN.BNN import BayesianNeuralNetwork

from variables import *
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn, save, load
from torchvision.transforms import ToTensor

#%% import files
TRAINDATASET = f'data/{DATASET}/min-max/train'
TESTDATASET = f'data/{DATASET}/min-max/test'

file_paths = glob.glob(os.path.join(project_path, TESTDATASET, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort() 

file_paths = file_paths[0:100]

x_test = [ToTensor()(np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)).to(device) for file_path in file_paths]
x_test = ToTensor()(np.genfromtxt(file_paths[0], delimiter=" ", dtype=np.float32)).to(device)

#convert to correct format
# from BNN.Data_loader import CustomDataset
# train = CustomDataset(TRAINDATASET)
# test = CustomDataset(TESTDATASET)

# train_set = DataLoader(train)
# test_set = DataLoader(test)

# Model input parameters
input_size = 14
hidden_size = 32
num_layers = 1

BNNmodel = BayesianNeuralNetwork(input_size, hidden_size, num_layers).to(device)
with open(f'{project_path}/BNN/model_states/BNN_model_state_{DATASET}_test.pt', 'rb') as f: 
    BNNmodel.load_state_dict(load(f)) 

background = []


explainer = shap.DeepExplainer(BNNmodel, x_test)
shap_values = explainer.shap_values(x_test)

shap.plots.waterfall(shap_values[0])