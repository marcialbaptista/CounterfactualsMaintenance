#%%Vizualizing script for the ML models
import os
import glob
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from torch import nn, load
from torchvision.transforms import ToTensor

from DNN import NeuralNetwork

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

device = 'cpu'

#import data
DATASET = 'FD001'
folder_path = f'data/{DATASET}/min-max/test'  # Specify the path to your folder

with open(os.path.join(project_path, folder_path, '0-Number_of_samples.csv')) as csvfile:
    sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory

file_paths = glob.glob(os.path.join(project_path, folder_path, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort() 

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
    for file_path in selected_file_paths:
        # Process each selected file
        sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
        label = float(file_path[-7:-4])

        #Import into trained machine learning models
        NNmodel = NeuralNetwork(input_size, hidden_size).to(device)
        with open(f'{project_path}/BNN/model_states/DNN_model_state_{DATASET}_test.pt', 'rb') as f: 
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

    #save engine results to file
    results = {
        'pred': y_pred_lst
    }

    save_to = os.path.join(project_path, 'BNN/DNN_results', DATASET)
    if not os.path.exists(save_to): os.makedirs(save_to)
    file_name = os.path.join(save_to, "result_{0:0=3d}.json".format(engine))
    
    with open(file_name, 'w') as jsonfile:
        json.dump(results, jsonfile)

#     plt.plot(y_pred_lst, label= 'Predicted RUL values')
#     plt.plot(y_lst, label='True RUL values')
    
# plt.xlabel('Cycles')
# plt.ylabel('RUL')
# plt.title(f'Dataset {DATASET}, RMSE = {D_RMSE}')
# plt.legend()
# plt.show()