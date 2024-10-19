#%% import dependencies
import os
import glob
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
import csv
from sklearn.model_selection import KFold

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

from torch import nn, save, load
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pickle

from variables import *


torch.manual_seed(42)
current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory


TRAIN = False
CV = False #Cross validation, if Train = True and CV = False, the model will train on the entire train data-set
SAVE = True
NOISY = True

noisy = 'noisy' if NOISY else 'denoised'

TRAINDATASET = f'data/{DATASET}/min-max/{noisy}/train'
TESTDATASET = f'data/{DATASET}/min-max/{noisy}/test'


#Frequentist neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=14, hidden_size=32, num_layers=1):
        super(NeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_size, 16), 
            # nn.ReLU(),
            nn.Linear(16,1),
            # nn.ReLU(),
        )
        
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial cell state
        
        out, _ = self.lstm(x)#, (h0, c0))
        out = out[:, -1, :]  # Extract the last time step output
        
        out = self.dense_layers(out) #pass through dense layers
        
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

        pred = model(X)

        ce_loss = torch.sqrt(loss_fn(pred[:,0], y)) #RMSE loss function
        
        #Backprop
        opt.zero_grad()
        ce_loss.backward()
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


        pred = model(X)

        ce_loss = torch.sqrt(loss_fn(pred[:,0], y)) #RMSE loss function
        loss_lst.append(ce_loss.item())

        test_loss = np.mean(loss_lst)

        if val:
            loop.set_description(f"Validate: {epoch}/{EPOCHS}")
            loop.set_postfix(val_loss = test_loss) 
        else:
            loop.set_description(f"Test: {epoch}/{EPOCHS}")
            loop.set_postfix(test_loss = test_loss) 
   
    return test_loss


#%% Training
if __name__ == "__main__":

    #training and validation data
    # %%Create an instance of the custom dataset
    from Data_loader import CustomDataset
    train = CustomDataset([TRAINDATASET])
    test = CustomDataset([TESTDATASET])

    # Model input parameters
    input_size = 14 #number of features
    hidden_size = 32 #numer of hidden cells in LSTM
    num_layers = 1 #number of LSTM layers

    #Model, optimizer and loss function
    NNmodel = NeuralNetwork(input_size, hidden_size, num_layers).to(device)
    opt = Adam(NNmodel.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Define the lambda function for decaying the learning rate
    lr_lambda = lambda epoch: 1 - (min(int(0.6*EPOCHS), epoch) / int(0.6*EPOCHS)) * (1 - 0.7) #after 60% of epochs reach 70% of learning rate
    # Create the learning rate scheduler
    scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

    #early stopping class that stops training to prevent overfitting
    from EarlyStopping import  EarlyStopping
    es = EarlyStopping()

    #%% Train the model
    loss_lst = []
    if TRAIN == True and CV == False: #Train the model on the entire data set
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

            train_loss = train_epoch(train_data=train_data, model=NNmodel, loss_fn=loss_fn, opt=opt)
            val_loss = test_epoch(test_data=val_data, model=NNmodel, loss_fn=loss_fn, val=True)

            train_loss_lst.append(train_loss) 
            val_loss_lst.append(val_loss) 
            

            if es(model=NNmodel, val_loss=val_loss): done = True #checks for validation loss threshold

        with open(f'BNN/model_states/DNN_model_state_{DATASET}_{noisy}.pt', 'wb') as f:
            save(NNmodel.state_dict(), f)

        # with open(f'BNN/model_states/DNN_model_state_{DATASET}_test.pkl', 'wb') as f:
        #     pickle.dump(NNmodel.state_dict(), f)

        plt.plot(loss_lst)
        plt.show()

    elif CV == True: #Perfrom Cross Validation
        splits = KFold(n_splits=k)
        history = {'Fold': [], 'Train loss': [], 'Test loss': []}
        total_set = ConcatDataset([train, test]) #for cross validation we look at the entire data set

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
            

            NNmodel = NeuralNetwork(input_size, hidden_size, num_layers).to(device)
            opt = Adam(NNmodel.parameters(), lr=1e-3)

            # Create the learning rate scheduler
            scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

            es = EarlyStopping()

            epoch = 0
            done = False
            while epoch < EPOCHS and not done:
                epoch +=1
                train_loss = train_epoch(train_data=train_data, model=NNmodel, loss_fn=loss_fn, opt=opt)
                val_loss = test_epoch(test_data=val_data, model=NNmodel, loss_fn=loss_fn, val=True)
                
                if es(model=NNmodel, val_loss=val_loss): done = True #checks for validation loss threshold

            test_loss = test_epoch(test_data=test_data, model=NNmodel, loss_fn=loss_fn)
            history['Test loss'].append(test_loss)
            history['Train loss'].append(train_loss)
            history['Fold'].append(fold)

            print(f'Fold {fold}: Training loss = {train_loss}, Test loss = {test_loss}')

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

                

    #%% Test the model and save files
    else:
        folder_path = f'data/{DATASET}/min-max/{noisy}/test_eval'  # Specify the path to your folder
        with open(os.path.join(project_path, folder_path, '0-Number_of_samples.csv')) as csvfile:
            sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory

        file_paths = glob.glob(os.path.join(project_path, folder_path, '*.txt'))  # Get a list of all file paths in the folder
        file_paths.sort() 

        RMSE_lst = []

        engines = np.arange(len(sample_len))
        for engine in engines:
            index = sum([int(sample_len[0:i+1][i][0]) for i in range(engine)])
            selected_file_paths = file_paths[index:index + int(sample_len[engine][0])]  # Select the desired number of files

            #setup data to plot
            y_pred_lst = []
            y_lst = []

            # Model input parameters
            input_size = 14 #number of features
            hidden_size = 32
            num_layers = 1

            #%%Go through each sample
            #Go through each sample
            loop = tqdm(selected_file_paths)
            for file_path in loop:
                # Process each selected file
                sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
                label = float(file_path[-7:-4])

                #Import into trained machine learning models
                NNmodel = NeuralNetwork(input_size, hidden_size).to(device)
                with open(f'{project_path}/BNN/model_states/DNN_model_state_{DATASET}_{noisy}.pt', 'rb') as f: 
                    NNmodel.load_state_dict(load(f)) 

                #predict RUL from samples
                X = ToTensor()(sample).to(device)
                y_pred = NNmodel(X)
            
                y_pred = y_pred[0].to('cpu')
                y_pred = y_pred.detach().numpy()

                y = label #True RUL

                #add predictions and true labels to lists
                y_pred_lst.append(y_pred.item())
                y_lst.append(y)
            

            error = [(y_pred_lst[i] - y_lst[i])**2 for i in range(len(y_lst))]
            D_RMSE = np.round(np.sqrt(np.mean(error)), 2)
            RMSE_lst.append(D_RMSE)

            #save engine results to file
            if SAVE:
                results = {
                    'pred': y_pred_lst,
                    'RMSE': D_RMSE
                }

                save_to = os.path.join(project_path, 'BNN/DNN_results', DATASET, noisy)
                if not os.path.exists(save_to): os.makedirs(save_to)
                file_name = os.path.join(save_to, "result_{0:0=3d}.json".format(engine))
                
                with open(file_name, 'w') as jsonfile:
                    json.dump(results, jsonfile)

        STD = np.sqrt(1/(len(engines) - 1) * sum([(RMSE_lst[i] - np.mean(RMSE_lst))**2 for i in range(len(RMSE_lst))]))
        COV = STD/np.mean(RMSE_lst)
        print(f'Evaluation completed for dataset {DATASET}')
        print(f'Deterministic Neural Network RMSE for {len(engines)} engines = {np.mean(RMSE_lst)} cycles')
        print(f'STD for RMSE: {STD}')
        print(f'COV for RMSE: {COV}')
    # %%

