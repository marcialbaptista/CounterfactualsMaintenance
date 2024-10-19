#%%Vizualizing script for the ML models
import os
import glob
import sys

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

import csv
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
from joblib import load

from variables import *

start = time.time()

folder_path = f'data/{DATASET}/min-max/test'  # Specify the path to your folder

with open(os.path.join(project_path, folder_path, '0-Number_of_samples.csv')) as csvfile:
    sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory

file_paths = glob.glob(os.path.join(folder_path, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort() 

#Import trained machine learning models
RVM_model = load(os.path.join(project_path, 'RVM/RVM_model.joblib'))

engines = [10]
for engine in engines:
    index = sum([int(sample_len[0:i+1][i][0]) for i in range(engine)])
    selected_file_paths = file_paths[index:index + int(sample_len[engine][0])]  # Select the desired number of files

    #setup data to plot
    pred_lst = []
    true_lst = []
    std_pred_lst = []

    #%%Go through each sample
    loop = tqdm(selected_file_paths)
    for file_path in loop:
    
        # Process each selected file
        sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
        sample = np.array([[element for row in sample for element in row]])
        label = float(file_path[-7:-4])
        

        prediction, std = RVM_model.predict(sample, return_std=True)
        
        y = label #True RUL

        #add predictions and true labels to lists
        pred_lst.append(prediction)
        true_lst.append(y)
        std_pred_lst.append(std)
        
        loop.set_description(f"Processing engine {engine}")


    error = [(pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))]
    B_RMSE = np.round(np.sqrt(np.mean(error)), 2)

    plt.plot(pred_lst, label= f'Bayesian Mean Predicted RUL values for engine {engine}, RMSE = {B_RMSE}')
    # plt.plot(y_pred_lst, label=f'Deterministic Predicted RUL values, RMSE = {D_RMSE}')
    plt.plot(true_lst, label='True RUL values')
    plt.fill_between(x=np.arange(len(pred_lst)), 
                     y1= pred_lst + std_pred_lst, 
                     y2=pred_lst - std_pred_lst,
                     alpha= 0.5,
                    #  color = 'yellow',
                     label= '1 STD interval'
                     )
    plt.fill_between(x=np.arange(len(pred_lst)), 
                     y1= pred_lst + 2*std_pred_lst, 
                     y2=pred_lst - 2*std_pred_lst,
                     alpha= 0.3,
                    #  color = 'yellow',
                     label= '2 STD interval'
                     )
#%%

finish = time.time()
print(f'elapsed time = {finish - start} seconds')
plt.xlabel('Cycles')
plt.ylabel('RUL')
plt.grid()
plt.title(f'Dataset {DATASET}, average variance = {np.round(np.mean(std_pred_lst**2),2)}')
plt.legend()
plt.show()