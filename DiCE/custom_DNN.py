#%% import dependencies
import os
import glob
import sys
import numpy as np
import torch

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

from torch import nn, save, load


from variables import *

#Frequentist neural network class
class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_size=14, hidden_size=32, num_layers=1):
        super(CustomNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_size, 16), 
            nn.Softplus(),
            nn.Linear(16,1),
            nn.Softplus(),
        )
        
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial cell state
        
        x = x.reshape(np.shape(x)[0],30,14)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Extract the last time step output
        
        out = self.dense_layers(out) #pass through dense layers
        
        return out