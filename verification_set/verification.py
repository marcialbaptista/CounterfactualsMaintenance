#Runs the models to verify their workings
import os
import sys
import torch
from torch import nn, save, load
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import glob

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory
sys.path.append(current_directory)

from BNN.BNN import BayesianNeuralNetwork
from BNN.Data_loader import CustomDataset

torch.manual_seed(42)

#Training loop per epoch
def train_epoch(train_data, model, loss_fn, opt):
    loss_lst = []
    
    for batch in train_data:
        X, y = batch #Input sample, true RUL
        y = torch.t(y) #Transpose to fit X dimension
       
        X, y = X.to(device), y.to(device) #send to device
        y_pred = model(X) #Run model

        kl = get_kl_loss(model) #Kullback Leibler loss
        ce_loss = torch.sqrt(loss_fn(y_pred[0][:,0], y)) #RMSE loss function
        loss = ce_loss + kl / BATCHSIZE #Loss including the KL loss
        loss_lst.append(ce_loss.item())
        # print(y_pred, y_pred[:,0],y, loss)
        
    
        #Backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
    
        train_loss = np.average(loss_lst)
        lr = opt.param_groups[0]['lr']
    
    return train_loss, lr



device = 'cpu'
DATASET = 'Verification'
TRAINDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/verification_set/train'))
TESTDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/verification_set/test'))
BATCHSIZE = 10
EPOCHS = 100

TRAIN = False

train = CustomDataset(TRAINDATASET)
test = CustomDataset(TESTDATASET)


# Import into trained machine learning models
input_size = 14
hidden_size = 32
num_layers = 1

model = BayesianNeuralNetwork(input_size, hidden_size, num_layers, prior_variance=1.0).to(device)
opt = Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

# Define the lambda function for decaying the learning rate
lr_lambda = lambda epoch: 1 - (min(int(0.6*EPOCHS), epoch) / int(0.6*EPOCHS)) * (1 - 0.1)
# Create the learning rate schedule
scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

    #%% Train the model
    
if TRAIN == True:

    print(f"Training model: {DATASET}")
    print(f"Batch size: {BATCHSIZE}, Epochs: {EPOCHS}")

    train_data = DataLoader(train, batch_size=BATCHSIZE)
    loss_lst = []
    for epoch in range(EPOCHS):

        train_loss, lr = train_epoch(train_data=train_data, model=model, loss_fn=loss_fn, opt=opt)
        
        scheduler.step() 
        loss_lst.append(train_loss) 
        print(f'Loss epoch {epoch} = {train_loss}, learning rate = {lr}') 

    with open(f'verification_set/verification_state.pt', 'wb') as f:
        save(model.state_dict(), f)

    plt.plot(loss_lst)
    plt.show()

else:
    #%%Vizualizing script for the ML models

    file_paths = glob.glob(os.path.join(TESTDATASET, '*.txt'))  # Get a list of all file paths in the folder
    file_paths.sort() 

    #setup data to plot
    mean_pred_lst = []
    true_lst = []
    var_pred_lst = []

    # Model input parameters
    input_size = 14 #number of features
    hidden_size = 32
    num_layers = 1

    index = 0
    for file_path in file_paths:
        print(f'Processing sample {index}')
        index += 1

        # Process each selected file
        sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
        label = float(file_path[-7:-4])

        #Import into trained machine learning models
        model = BayesianNeuralNetwork(input_size, hidden_size, num_layers).to(device)
        with open(f'verification_set/verification_state.pt', 'rb') as f: 
            model.load_state_dict(load(f)) 

        #predict RUL from samples using Monte Carlo Sampling
        X = ToTensor()(sample).to(device)
        n_samples = 10

        mc_pred = [model(X)[0] for _ in range(n_samples)]


        predictions = torch.stack(mc_pred)
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        y = label #True RUL

        #add predictions and true labels to lists
        mean_pred_lst.append(mean_pred.item())
        true_lst.append(y)
        var_pred_lst.append(var_pred.item())


    error = [(mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))]
    B_RMSE = np.round(np.sqrt(np.mean(error)), 2)

    plt.plot(mean_pred_lst, label= f'Bayesian Mean Predicted values, RMSE = {B_RMSE}')
    plt.plot(true_lst, label='True values')
    plt.fill_between(x=np.arange(len(mean_pred_lst)), 
                    y1= mean_pred_lst + np.sqrt(var_pred_lst), 
                    y2=mean_pred_lst - np.sqrt(var_pred_lst),
                    alpha= 0.5,
                    label= '1 STD interval'
                    )
    plt.fill_between(x=np.arange(len(mean_pred_lst)), 
                    y1= mean_pred_lst + 2*np.sqrt(var_pred_lst), 
                    y2=mean_pred_lst - 2*np.sqrt(var_pred_lst),
                    alpha= 0.3,
                    label= '2 STD interval'
                    )
    #%%
    plt.xlabel('Sample')
    plt.ylabel('Output')
    plt.title(f'Dataset {DATASET}, {n_samples} samples per data point, average variance = {np.round(np.mean(var_pred_lst),2)}')
    plt.legend()
    plt.show()