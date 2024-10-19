#%% import dependencies
import glob
import sys
import os
import multiprocessing as mp

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
import numpy as np
from torch import nn
import bayesian_torch.layers as bl

# from variables import *
torch.manual_seed(42)

device = 'cpu' #device where models whill be run


#Bayesian neural network class
class CustomBayesianNeuralNetwork(nn.Module):
    """Bayesian Neural Network using LSTM and linear layers. Deterministic to Bayesian using Reparameterization.

    Args:
        input_size: number of input features
        hidden_szie: size of hidden node vector (also size of output)
        num_layers: amount of LSTM layers
        prior_mean: initial guess for parameter mean
        prior_variance: initial guess for parameter variance
        posterior_mu_init: init std for the trainable mu parameter, sampled from N(0, posterior_mu_init)
        posterior_rho_init: init std for the trainable rho parameter, sampled from N(0, posterior_rho_init)

    """
    def __init__(self, loop_size = 10, input_size=14, hidden_size=32, num_layers=1, prior_mean = 0.0, prior_variance = 1.0, posterior_mu_init = 0.0, posterior_rho_init = -3.0):
        super(CustomBayesianNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loop_size = loop_size
        self.lstm = bl.LSTMReparameterization(in_features= input_size, out_features= hidden_size, prior_mean=prior_mean, prior_variance=prior_variance, posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init)
        self.relu = bl.ReLU()
        self.l1 = bl.LinearReparameterization(in_features=hidden_size, out_features=16)
        self.l2 = bl.LinearReparameterization(16,1)
        
        
    def loop_forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial hidden state
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial cell state

        x = x.reshape(np.shape(x)[0],30,14)
        # x = np.random.rand(1,30,14)

        out = self.lstm(x)#, (h0, c0))
        
        out = out[0][:, -1, :]  # Extract the last time step output
        
        out = self.l1(out) #pass through dense layers
       
        out = self.l2(out[0])
    
        return out[0]
    
    def forward(self, x):
        
        mc_pred = [self.loop_forward(x) for _ in range(self.loop_size)]
        # print(mc_pred)
        predictions = torch.stack(mc_pred)
        # print(predictions.tolist())
        mean_pred = torch.mean(predictions, dim=0)      
        

        return mean_pred