#%% import dependencies
import glob
import sys
import os
import csv
import json

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

import numpy as np
import torch
import tqdm
from sklearn.model_selection import KFold
import pandas as pd

from torch import nn, save, load
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import multiprocessing as mp

import bayesian_torch.layers as bl

from BNN.BNN import BayesianNeuralNetwork

from variables import *
torch.manual_seed(42)


SAVE = True #if true, result will be saved to json files
NOISY = False
INCREASE = False

noisy = 'noisy' if NOISY else 'denoised'
increase = 'increase' if INCREASE else 'decrease'

CF_DATASET = f'DiCE/BNN_cf_results/inputs/{DATASET}/{increase}/{noisy}'
folder_path = f'data/{DATASET}/min-max/{noisy}/test_eval'  # Specify the path to your input folder

with open(os.path.join(project_path, folder_path, '0-Number_of_samples.csv')) as csvfile:
        sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory


def open_cf(file_path):
    cf_data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row_number, row in enumerate(csv_reader):
            if row_number == 1:  # Skip the sensor name row
                if row[0] != '':
                    cf_data = [np.float32(value) for value in row]

                    # Step 2: Remove the final entry
                    cf_RUL = cf_data[-1]
                    cf_data = cf_data[:-1]

                    # Step 3: Convert the modified second row into a 2D NumPy array
                    shape = (30, 14)  # Desired shape
                    array = np.array(cf_data).reshape(shape)
                
                else:
                    array, cf_RUL = no_cf(file_path)



    return array, cf_RUL

def no_cf(file_path):
    file_id = file_path[-13:-4]
    file_orig = f'{folder_path}/test_eval_{file_id[:-4]}-{file_id[-3:]}.txt'

    sample = np.genfromtxt(file_orig, delimiter=" ", dtype=np.float32)
    label = float(file_orig[-7:-4])

    return sample, label
      

# Function to split a list into chunks
def chunk_list(input_list, num_chunks):
    avg_chunk_size = len(input_list) // num_chunks
    remainder = len(input_list) % num_chunks

    chunks = []
    start = 0

    for i in range(num_chunks):
        chunk_size = avg_chunk_size + (1 if i < remainder else 0)
        end = start + chunk_size
        chunks.append(input_list[start:end])
        start = end

    return chunks

def CF_results(chunk):
    file_paths = glob.glob(os.path.join(CF_DATASET, '*.csv'))  # Get a list of all counterfactual input files
    file_paths.sort() 

    engines = chunk

    # Model input parameters
    input_size = 14 #number of features
    hidden_size = 32
    num_layers = 1

    #Import into trained machine learning models
    NNmodel = BayesianNeuralNetwork(input_size, hidden_size).to(device)
    with open(f'{project_path}/BNN/model_states/BNN_model_state_{DATASET}_{noisy}_orig.pt', 'rb') as f: 
        NNmodel.load_state_dict(load(f)) 

    for engine in engines:
        index = sum([int(sample_len[0:i+1][i][0]) for i in range(engine)])
        selected_file_paths = file_paths[index:index + int(sample_len[engine][0])]  # Select the desired number of files
        # selected_file_paths = file_paths[0:170]

        #setup data to plot
        mean_pred_lst = []
        true_lst = []
        var_pred_lst = []


        #Go through each sample
        for file_path in selected_file_paths:
        
            # Process each selected file
            sample, cf_RUL = open_cf(file_path)
            label = float(file_path[-7:-4])


            #predict RUL from samples using Monte Carlo Sampling
            X = ToTensor()(sample).to(device)
            n_samples = 10

            mc_pred = [NNmodel(X)[0] for _ in range(n_samples)]


            predictions = torch.stack(mc_pred)
            mean_pred = torch.mean(predictions, dim=0)
            # print(mean_pred, cf_RUL)
            var_pred = torch.var(predictions, dim=0)
            y = label #True RUL

            #add predictions and true labels to lists
            mean_pred_lst.append(mean_pred.item())
            true_lst.append(y)
            var_pred_lst.append(var_pred.item())

        error = [(mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))] #squared BNN error
        B_RMSE = np.round(np.sqrt(np.mean(error)), 2) #Root Mean Squared error of Bayesian prediciton

        #save engine results to file
        if SAVE:
            results = {
                'mean': mean_pred_lst,
                'var': var_pred_lst,
                'true': true_lst,
                'RMSE': B_RMSE
            }

            save_to = os.path.join(project_path, 'DiCE/BNN_cf_results/outputs', DATASET, increase, noisy)
            if not os.path.exists(save_to): os.makedirs(save_to)
            file_name = os.path.join(save_to, "cf_result_{0:0=3d}.json".format(engine))
            
            with open(file_name, 'w') as jsonfile:
                json.dump(results, jsonfile)



if __name__ == "__main__":
    
    
    num_cores = mp.cpu_count()

    engines = np.arange(len(sample_len))
    # engines = [0]

    chunks = chunk_list(engines, min(num_cores, len(engines)))


    with mp.Pool(processes=min(num_cores, len(chunks))) as pool:
        list(tqdm.tqdm(pool.imap_unordered(CF_results, chunks), total=len(chunks)))