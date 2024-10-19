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
import argparse

from bayesian_torch.models.dnn_to_bnn import get_kl_loss
import bayesian_torch.layers as bl

import scipy.stats as stats

from variables import *


torch.manual_seed(42)
current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory

parser = argparse.ArgumentParser(description="Script to train, evaluate and retrain BNN model")

parser.add_argument('--TRAIN', action='store_true', default=False, help="If train = True, the model will either train or perform cross-validation.")
parser.add_argument('--CV', action='store_true', default=False, help="Cross-validation. If Train = True and CV = False, the model will train on the entire train dataset.")
parser.add_argument('--SAVE', action='store_true', default=True, help="If True, will save BNN output to .json files.")
parser.add_argument('--NOISY', action='store_true', default=True, help="If True, use noisy (normalized) data.")

parser.add_argument('--TEST_SET', action='store_true', default=False, help="Uses the provided test set of CMAPSS instead of the test-train split.")
parser.add_argument('--CF_TRAIN', action='store_true', default=False, help="If true, counterfactuals will be added to the training data.")
parser.add_argument('--NOCF_TRAIN', action='store_true', default=False, help="If true, non cf converted inputs will be added to the training data (unless CF_TRAIN = True).")

parser.add_argument('--EVAL', action='store_true', default=False, help="If true, the eval test set will be saved. If false, the normal test set will be saved (to be converted to counterfactuals).")
parser.add_argument('--CHECK_DIST', action='store_true', default=False, help="If True, output distribution will be plotted using a QQ plot.")

args = parser.parse_args()

# TRAIN = False #If train = True, the model will either train or perfrom cross validation, if both TRAIN and CV = False, the model will run and save results
# CV = False #Cross validation, if Train = True and CV = False, the model will train on the entire train data-set
# SAVE = False #If True, will save BNN output to .json files
# NOISY = True #If True, use noisy (normalized) data

# TEST_SET = False #Uses the provided test set of CMAPSS instead of test-train split
# CF_TRAIN = False #If true, counterfatuals will be added to the training data
# NOCF_TRAIN = False #If true, non cf converted inputs will be added to the training data (unless CF_TRAIN = True)

# EVAL = True #If true, the eval test set will be saved. If false, the normal test set will be saved (to be converted to counterfactuals)
# CHECK_DIST = True #If True, output distribution will be plotted using a QQ plot

noisy = 'noisy' if args.NOISY else 'denoised'
cf = 'CF' if args.CF_TRAIN else ('NOCF' if args.NOCF_TRAIN else 'orig')
eval = 'test_eval' if args.EVAL else 'test'

TRAINDATASET = f'data/{DATASET}/min-max/{noisy}/train'
TESTDATASET = f'data/{DATASET}/min-max/{noisy}/test'
EVALDATASET = f'data/{DATASET}/min-max/{noisy}/test_eval'
CFDATASET = f'DiCE_uncertainty/BNN_cf_results/inputs/{DATASET}/{noisy}'

if args.TEST_SET:
    test_path = f'{DATASET}/{noisy}/test_set'
else:
    test_path = f'{DATASET}'

#Bayesian neural network class
class BayesianNeuralNetwork(nn.Module):
    """Bayesian Neural Network using LSTM and linear layers. Deterministic to Bayesian using Reparameterization.

    Args:
        input_size: number of input features
        hidden_szie: size of hidden node vector (also size of output)
        num_layers: amountof LSTM layers
        prior_mean: initial guess for parameter mean
        prior_variance: initial guess for parameter variance
        posterior_mu_init: init std for the trainable mu parameter, sampled from N(0, posterior_mu_init)
        posterior_rho_init: init std for the trainable rho parameter, sampled from N(0, posterior_rho_init)

    """
    def __init__(self, input_size=14, hidden_size=32, num_layers=1, prior_mean = 0.0, prior_variance = 1.0, posterior_mu_init = 0.0, posterior_rho_init = -3.0):
        super(BayesianNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = bl.LSTMReparameterization(in_features= input_size, out_features= hidden_size, prior_mean=prior_mean, prior_variance=prior_variance, posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init)
        self.relu = bl.ReLU() 
        self.l1 = bl.LinearReparameterization(in_features=hidden_size, out_features=16)
        self.l2 = bl.LinearReparameterization(16,1)
        
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial cell state
        
        out = self.lstm(x)#, (h0, c0))
        
        out = out[0][:, -1, :]  # Extract the last time step output
        
        out = self.l1(out) #pass through dense layers

        out = self.relu(out)
       
        out = self.l2(out[0])

        out = self.relu(out)
    
        return out

#Training loop per epoch
def train_epoch(train_data, model, loss_fn, opt):
    """Trains the model over one epoch

    Args:
        train_data (array (torch)): Input time series data [30, 14]
        model (torch model): (neural network) pytorch model
        loss_fn (_type_): Loss function
        opt (_type_): Optimizer

    Returns:
        float : RMSE training loss at the end of an epoch
    """
    model.train()
    loop = tqdm(train_data)
    
    for batch in loop:
        X, y = batch #Input sample, true RUL
        y = torch.t(y) #Transpose to fit X dimension
       
        X, y = X.to(device), y.to(device) #send to device

        n_samples = 10

        mc_pred = [model(X)[0] for _ in range(n_samples)]

        predictions = torch.stack(mc_pred)
        mean_pred = torch.mean(predictions, dim=0)

        ce_loss = torch.sqrt(loss_fn(mean_pred[:,0], y)) #RMSE loss function

        kl = get_kl_loss(model) #Kullback Leibler loss
        loss = ce_loss + kl / BATCHSIZE #Loss including the KL loss
        
        #Backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
    
        train_loss = ce_loss.item()

        loop.set_description(f"Epoch: {epoch}/{EPOCHS}")
        loop.set_postfix(train_loss = train_loss)#, lr = opt.param_groups[0]['lr']) 

    scheduler.step()
    
    return train_loss

#Validation loop per epoch
def test_epoch(test_data, model, loss_fn, val = False):
    """Trains the model over one epoch

    Args:
        train_data (array (torch)): Input time series data [30, 14]
        model (torch model): (neural network) pytorch model
        loss_fn (_type_): Loss function
        val (bool): if True then used for validation

    Returns:
        float : average RMSE test loss over the entire epoch
    """
    model.eval()

    loop = tqdm(test_data)
    loss_lst = []
    for batch in loop:
        
        X, y = batch #Input sample, true RUL
        # y = torch.t(y) #Transpose to fit X dimension
        X, y = X.to(device), y.to(device) #send to device

        n_samples = 10

        mc_pred = [model(X)[0] for _ in range(n_samples)]

        predictions = torch.stack(mc_pred)
        mean_pred = torch.mean(predictions, dim=0)

        ce_loss = torch.sqrt(loss_fn(mean_pred[:,0], y)) #RMSE loss function
        loss_lst.append(ce_loss.item())

        test_loss = np.mean(loss_lst)

        if val:
            loop.set_description(f"Validate: {epoch}/{EPOCHS}")
            loop.set_postfix(val_loss = test_loss) 
        else:
            loop.set_description(f"Test: {epoch}/{EPOCHS}")
            loop.set_postfix(test_loss = test_loss) 
   
    return test_loss


# with open(f'BNN/model_state_{DATASET}.pt', 'rb') as f: 
#     NNmodel.load_state_dict(load(f)) 

#%% main script
if __name__ == '__main__':

    from Data_loader import CustomDataset
    
    test = CustomDataset([TESTDATASET])
    test_eval = CustomDataset([EVALDATASET])

    if args.CF_TRAIN:
        train = CustomDataset([TRAINDATASET, CFDATASET]) #include counterfactual inputs in the training data
    elif args.NOCF_TRAIN:
        train = CustomDataset([TRAINDATASET, TESTDATASET]) #include non-counterfactual (original) inputs in training data
    else:
        train = CustomDataset([TRAINDATASET])

    # Model input parameters
    input_size = 14
    hidden_size = 32
    num_layers = 1

    BNNmodel = BayesianNeuralNetwork(input_size, hidden_size, num_layers).to(device)
    opt = Adam(BNNmodel.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Define the lambda function for decaying the learning rate
    lr_lambda = lambda epoch: 1 - (min(int(0.6*EPOCHS), epoch) / int(0.6*EPOCHS)) * (1 - 0.7) #after 60% of epochs reach 70% of learning rate
    # Create the learning rate scheduler
    scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

    #early stopping class that stops training to prevent overfitting
    from EarlyStopping import  EarlyStopping
    es = EarlyStopping()

    #%% Train the model
    if args.TRAIN == True and args.CV == False:

        print(f"Training model: {DATASET}")
        print(f"Batch size: {BATCHSIZE}, Epochs: {EPOCHS}")

        train_set, val_set = random_split(train, [0.8, 0.2])

        train_data = DataLoader(train_set, batch_size=BATCHSIZE)
        val_data = DataLoader(val_set, batch_size= len(val_set))

        train_loss_lst = []
        val_loss_lst = []
        epoch = 0
        done = False
        while epoch < EPOCHS and not done:
            epoch +=1

            train_loss = train_epoch(train_data=train_data, model=BNNmodel, loss_fn=loss_fn, opt=opt)
            val_loss = test_epoch(test_data=val_data, model=BNNmodel, loss_fn=loss_fn, val=True)

            train_loss_lst.append(train_loss) 
            val_loss_lst.append(val_loss) 
            

            if es(model=BNNmodel, val_loss=val_loss): done = True #checks for validation loss threshold

        with open(f'BNN/model_states/BNN_model_state_{DATASET}_{noisy}_{cf}.pt', 'wb') as f:
            save(BNNmodel.state_dict(), f)

        # with open(f'BNN/model_states/BNN_model_state_{DATASET}_test.pkl', 'wb') as f:
        #     pickle.dump(BNNmodel.state_dict(), f)

        # plt.plot(train_loss_lst, label='Train loss')
        # plt.plot(val_loss_lst, label='Validation loss')
        # plt.legend()
        # plt.show()

    #%% Cross validation
    elif args.CV == True: #Perfrom Cross Validation
        splits = KFold(n_splits=k)
        history = {'Fold': [], 'Train loss': [], 'Test loss': []}
        total_set = ConcatDataset([train, test, test_eval]) #for cross validation we look at the entire data set

        train_test_set, val_set = random_split(total_set, [0.8, 0.2])

        for fold , (train_idx, test_idx) in enumerate(splits.split(np.arange(len(train_test_set)))):
            
            print(f'Fold {fold + 1}')

            # train_test_set, val_set = random_split(total_set, [0.8, 0.2])

            #Train and test data split according to amount of folds
            train_sampler = SequentialSampler(train_idx) #(k-1)/k part of the total set
            test_sampler = SequentialSampler(test_idx) #1/k part of the total set
            train_data = DataLoader(train_test_set, batch_size=BATCHSIZE, sampler=train_sampler)
            test_data = DataLoader(train_test_set, batch_size= BATCHSIZE, sampler=test_sampler)
            val_data = DataLoader(val_set, batch_size=len(val_set))
            

            BNNmodel = BayesianNeuralNetwork(input_size, hidden_size, num_layers).to(device)
            opt = Adam(BNNmodel.parameters(), lr=1e-3)

            # Create the learning rate scheduler
            scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

            es = EarlyStopping()

            epoch = 0
            done = False
            while epoch < EPOCHS and not done:
                epoch +=1
                train_loss = train_epoch(train_data=train_data, model=BNNmodel, loss_fn=loss_fn, opt=opt)
                val_loss = test_epoch(test_data=val_data, model=BNNmodel, loss_fn=loss_fn, val=True)
                
                if es(model=BNNmodel, val_loss=val_loss): done = True #checks for validation loss threshold

            test_loss = test_epoch(test_data=test_data, model=BNNmodel, loss_fn=loss_fn)
            history['Test loss'].append(test_loss)
            history['Train loss'].append(train_loss)
            history['Fold'].append(fold)

            print(f'Fold {fold + 1}: Training loss = {train_loss}, Test loss = {test_loss}')

        print(f'Performance of {k} fold cross validation')
        print(f'Average training loss: {np.mean(history["Train loss"])}')
        print(f'Average test loss: {np.mean(history["Test loss"])}')

        #%% plot training and testing loss per fold
        df = pd.DataFrame(history)
        df.plot(x = 'Fold', y = ['Train loss', 'Test loss'], kind='bar')
        plt.xlabel('Fold')
        plt.ylabel('Loss')
        plt.title(f'{k} fold cross validation')
        plt.legend()

        plt.show()
    #%% Test the model and save results
    else:
        folder_path = f'data/{test_path}/min-max/{noisy}/{eval}'  # Specify the path to your folder

        with open(os.path.join(project_path, folder_path, '0-Number_of_samples.csv')) as csvfile:
            sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory

        file_paths = glob.glob(os.path.join(folder_path, '*.txt'))  # Get a list of all file paths in the folder
        file_paths.sort() 

        RMSE_lst = [] #Overal RMSE list of all engines
        mean_preds = [] #overall means of all engines
        pred_dist_lst = [] #List containing all prediciton distributions

        var_dict = {} #dictionary with key: sample id, value: variance. Will be used in DiCE_uncertianty

        # from DNN import RMSE_lst as DRMSE_lst
       
        engines = np.arange(len(sample_len))
        for engine in engines:
            index = sum([int(sample_len[0:i+1][i][0]) for i in range(engine)])
            selected_file_paths = file_paths[index:index + int(sample_len[engine][0])]  # Select the desired number of files
            # selected_file_paths = file_paths[0:1]

            #setup data to save
            mean_pred_lst = []
            true_lst = []
            var_pred_lst = []

            # Model input parameters
            input_size = 14 #number of features
            hidden_size = 32
            num_layers = 1

            #Go through each sample
            loop = tqdm(selected_file_paths)
            for file_path in loop:
            
                # Process each selected file
                sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
                sample_id = int(file_path[-13:-8])
                label = float(file_path[-7:-4])

                #Import into trained machine learning models
                NNmodel = BayesianNeuralNetwork(input_size, hidden_size).to(device)
                with open(f'{project_path}/BNN/model_states/BNN_model_state_{DATASET}_{noisy}_{cf}.pt', 'rb') as f: 
                    NNmodel.load_state_dict(load(f)) 

                #predict RUL from samples using Monte Carlo Sampling
                X = ToTensor()(sample).to(device)
                n_samples = 10
                NNmodel.eval()

                mc_pred = [NNmodel(X)[0] for _ in range(n_samples)]

                # if sample_id == 100:
                #     plt.hist([mc_pred[i].item() for i in range(n_samples)], bins=50)
                #     plt.show()


                predictions = torch.stack(mc_pred)
                mean_pred = torch.mean(predictions, dim=0)
                # print(mean_pred)
                var_pred = torch.var(predictions, dim=0)
                y = label #True RUL

                #add predictions and true labels to lists
                mean_pred_lst.append(mean_pred.item())
                mean_preds.append(mean_pred.item())
                true_lst.append(y)
                var_pred_lst.append(var_pred.item())
                var_dict[int(sample_id)] = var_pred.item()
                
                loop.set_description(f"Processing engine {engine}")

            error = [(mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))] #squared BNN error
            B_RMSE = np.round(np.sqrt(np.mean(error)), 2) #Root Mean Squared error of Bayesian prediciton
            RMSE_lst.append(B_RMSE)

            if args.CHECK_DIST:
                pred_dist = [predictions.detach().numpy()[i][0][0] for i in range(len(predictions.detach().numpy()))]
                pred_dist_lst.append(pred_dist)

            #save engine results to file
            if args.SAVE:
                results = {
                    'mean': mean_pred_lst,
                    'var': var_pred_lst,
                    'true': true_lst,
                    'RMSE': B_RMSE
                }

                save_to = os.path.join(project_path, 'BNN/BNN_results', test_path, f'{noisy}-{cf}-test')
                if not os.path.exists(save_to): os.makedirs(save_to)
                file_name = os.path.join(save_to, "result_{0:0=3d}.json".format(engine))
                
                with open(file_name, 'w') as jsonfile:
                    json.dump(results, jsonfile)

        #save variance dictionary to file to be used in DiCE_uncertainty
        if args.SAVE:
            save_to = os.path.join(project_path, 'DiCE_uncertainty/BNN_results', DATASET, f'{noisy}-{cf}-test')
            if not os.path.exists(save_to): os.makedirs(save_to)
            file_name = os.path.join(save_to, f"variance_results-{eval}.json")
            
            with open(file_name, 'w') as jsonfile:
                json.dump(var_dict, jsonfile)

        STD = np.sqrt(1/(len(engines) - 1) * sum([(RMSE_lst[i] - np.mean(RMSE_lst))**2 for i in range(len(RMSE_lst))]))
        COV = STD/np.mean(RMSE_lst)
        print(f'Evaluation completed for dataset {DATASET}. Noisy: {args.NOISY}. Counterfactuals: {cf}')
        print(f'Bayesian Neural Network RMSE for {len(engines)} engines = {np.mean(RMSE_lst)} cycles')
        print(f'STD for RMSE: {STD}')
        print(f'COV for RMSE: {COV}')

    if args.CHECK_DIST:
        # List of distribution types to compare against
        distribution_types = ['norm', 'expon', 'uniform', 'gamma', 'poisson']

        # Create a figure with subplots
        num_subplots = len(distribution_types)
        fig, axs = plt.subplots(1, num_subplots, figsize=(15, 5))

        # Iterate through each distribution type and plot a Q-Q plot on a separate subplot
        for i, dist_type in enumerate(distribution_types):
            for pred_dist in pred_dist_lst:
                if dist_type == 'gamma':
                    shape, loc, scale = stats.gamma.fit(pred_dist)
                    stats.probplot(pred_dist, dist=dist_type, sparams=(shape,), plot=axs[i])
                elif dist_type == 'poisson':
                    mu = np.mean(pred_dist)
                    stats.probplot(pred_dist, dist=dist_type, sparams=(mu,), plot=axs[i])
                else:
                    stats.probplot(pred_dist, dist=dist_type, plot=axs[i])
                axs[i].set_title(f'Distribution: {dist_type}')

        # Adjust layout for better visualization
        plt.tight_layout()
        plt.show()

        

        # plt.plot(np.arange(len(RMSE_lst)), RMSE_lst, label="Bayesian")
        # plt.plot(np.arange(len(DRMSE_lst)), DRMSE_lst, label="Deterministic")
        # plt.xlabel('Engines')
        # plt.ylabel('RMSE')
        # plt.show()