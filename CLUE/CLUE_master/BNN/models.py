from __future__ import division
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from CLUE_master.src.layers import SkipConnection, MLPBlock
import bayesian_torch.layers as bl

from variables import *

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
    def __init__(self, loop_size = 10, input_size=14, hidden_size=32, num_layers=1, output_dim=1, prior_mean = 0.0, prior_variance = 1.0, posterior_mu_init = 0.0, posterior_rho_init = -3.0, grad=False):
        super(CustomBayesianNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loop_size = loop_size
        self.output_dim = output_dim
        self.lstm = bl.LSTMReparameterization(in_features= self.input_size, out_features= self.hidden_size, prior_mean=prior_mean, prior_variance=prior_variance, posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init)
        self.relu = bl.ReLU()
        self.l1 = bl.LinearReparameterization(in_features=self.hidden_size, out_features=16)
        self.l2 = bl.LinearReparameterization(16,1)
        
        
    def loop_forward(self, input):
        # h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device) #initial hidden state
        # c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device) #initial cell state

        
        out = self.lstm(input)#, (h0, c0))
        
        out = out[0][:, -1, :]  # Extract the last time step output
        
        out = self.l1(out) #pass through dense layers
       
        out = self.l2(out[0])
    
        return out[0]
    
    def forward(self, x):
        mu = torch.empty(0)
        sigma = torch.empty(0)
        
        # for i in range(len(x)):
        input = x.reshape(np.shape(x)[0],30,14)
        # input = np.random.rand(1,30,14)
        mc_pred = [self.loop_forward(input) for _ in range(self.loop_size)]
        # print(mc_pred)
        predictions = torch.stack(mc_pred)
        # print(predictions.tolist())
        mu = torch.cat((mu, torch.mean(predictions, dim=0)), dim=0)      
        sigma = torch.cat((sigma, torch.std(predictions, dim=0)), dim=0)

        
        return mu, sigma
    
class MLP_gauss(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, flatten_image):
        super(MLP_gauss, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.flatten_image = flatten_image

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 2*output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_image:
            x = x.view(-1, self.input_dim)
        x = self.block(x)
        mu = x[:, :self.output_dim]
        sigma = F.softplus(x[:, self.output_dim:])
        return mu, sigma

class MLP(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, flatten_image):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.flatten_image = flatten_image

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_image:
            x = x.view(-1, self.input_dim)
        return self.block(x)

class MLP_dirichlet(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, flatten_image):
        super(MLP_dirichlet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.flatten_image = flatten_image

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_image:
            x = x.view(-1, self.input_dim)
        x = self.block(x)
        dir_params = F.softplus(x)
        return dir_params

# CNN code

class MNIST_small_cnn(nn.Module):
    def __init__(self,):
        super(MNIST_small_cnn, self).__init__()

        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 600)
        self.fc2 = nn.Linear(600, self.output_dim)

        # choose your non linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        #         x = x.view(-1, self.input_dim) # view(batch_size, input_dim)
        x = self.conv1(x)
        x = self.act(x)
        # -----------------
        x = self.conv2(x)
        x = self.act(x)
        # -----------------
        x = x.view(-1, 7 * 7 * 64)
        # -----------------
        x = self.fc1(x)
        x = self.act(x)
        # -----------------
        y = self.fc2(x)

        return y


############################# Skip layer models


class MLP_skip(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, flatten_image):
        super(MLP_skip, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.flatten_image = flatten_image

        layers = [nn.Linear(input_dim, width), nn.ReLU(), nn.BatchNorm1d(num_features=width)]
        for i in range(depth - 1):
            layers.append(
                MLPBlock(width)
            )

        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_image:
            x = x.view(-1, self.input_dim)
        return self.block(x)


class MLP_skip_gauss(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, flatten_image):
        super(MLP_skip_gauss, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.flatten_image = flatten_image

        layers = [nn.Linear(input_dim, width), nn.ReLU(), nn.BatchNorm1d(num_features=width)]
        for i in range(depth - 1):
            layers.append(
                MLPBlock(width)
            )

        layers.append(nn.Linear(width, 2 * output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_image:
            x = x.view(-1, self.input_dim)
        x = self.block(x)
        mu = x[:, :self.output_dim]
        sigma = F.softplus(x[:, self.output_dim:])
        return mu, sigma


class MLP_skip_dirichlet(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, flatten_image):
        super(MLP_skip_dirichlet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.flatten_image = flatten_image

        layers = [nn.Linear(input_dim, width), nn.ReLU(), nn.BatchNorm1d(num_features=width)]
        for i in range(depth - 1):
            layers.append(
                MLPBlock(width)
            )

        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_image:
            x = x.view(-1, self.input_dim)
        x = self.block(x)
        dir_params = F.softplus(x)
        return dir_params


####### Dropout models


class MLP_drop(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, flatten_image):
        super(MLP_drop, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.flatten_image = flatten_image

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))

        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_image:
            x = x.view(-1, self.input_dim)
        return self.block(x)


class MLP_drop_gauss(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, flatten_image):
        super(MLP_drop_gauss, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.flatten_image = flatten_image

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))

        layers.append(nn.Linear(width, 2 * output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_image:
            x = x.view(-1, self.input_dim)
        x = self.block(x)
        mu = x[:, :self.output_dim]
        sigma = F.softplus(x[:, self.output_dim:])
        return mu, sigma


class MLP_drop_dirichlet(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, flatten_image):
        super(MLP_drop_dirichlet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.flatten_image = flatten_image

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))

        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_image:
            x = x.view(-1, self.input_dim)
        x = self.block(x)
        dir_params = F.softplus(x)
        return dir_params
