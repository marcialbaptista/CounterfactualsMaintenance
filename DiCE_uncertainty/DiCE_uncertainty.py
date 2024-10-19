#Script to extract counterfactual explanations using DiCE package
#%% import dependencies
import sys
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

import glob
import csv
import numpy as np
import pandas as pd
from torch import load
import multiprocessing as mp
import time
import tqdm
import json
import argparse

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

import dice_ml_custom as dice_ml_custom
# import dice_ml as dice_ml_custom

from custom_BNN import CustomBayesianNeuralNetwork

device = 'cpu' #device where models whill be run
DATASET = 'FD001' #which data set to use from cmpass [FD001, FD002, FD003, FD004]

BATCHSIZE = 100
EPOCHS = 100

parser = argparse.ArgumentParser(description="Script to train, evaluate and retrain BNN model")
parser.add_argument('--NOISY', action='store_true', default=False, help="If True, use noisy (normalized) data.")
args = parser.parse_args()
# NOISY = True

noisy = 'noisy' if args.NOISY else 'denoised'

#%% import files
TRAINDATASET = f'data/{DATASET}/min-max/{noisy}/train'
TESTDATASET = f'data/{DATASET}/min-max/{noisy}/test'

BayDet = 'BNN'

with open(os.path.join(project_path, TESTDATASET, '0-Number_of_samples.csv')) as csvfile:
    sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory
    sample_len = [int(sample_len[i][0]) for i in range(len(sample_len))] #convert to list of ints

#Import into trained machine learning models
if BayDet == 'BNN':
    NNmodel = CustomBayesianNeuralNetwork().to(device)


with open(f'{project_path}/BNN/model_states/{BayDet}_model_state_{DATASET}_{noisy}_orig.pt', 'rb') as f: 
    NNmodel.load_state_dict(load(f)) 

file_paths = glob.glob(os.path.join(project_path, TESTDATASET, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort()

#split test set into 2 distinct parts, 1 to convert to cfs and one to evaluate
test_to_cf = file_paths # 20% of total data set to be converted to cf



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
#%%Go over each sample
def CMAPSS_counterfactuals(chunk):

    var_dict = os.path.join(project_path, 'DiCE_uncertainty/BNN_results', DATASET, f'{noisy}-orig/variance_results-test.json')

    with open(var_dict, 'r') as jsonfile:
        var_dict = json.load(jsonfile)

    for file_path in chunk:
        #load sample with true RUL
        sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
        sample_id = int(file_path[-13:-8])
        label = int(file_path[-7:-4])
        std = np.sqrt(var_dict[str(sample_id)])

        #Create labels for sensors and RUL
        sensors = [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
        head = [[f'Sensor {i,j}' for j in range(len(sample)) for i in sensors]]
        head[0].append('STD')

        #Flatten sample and combine with RUL
        sample = [[element for row in sample for element in row]] #flatten time series sample into format [(sensor 1, timestep 0),...(sensor n, timestep w)]
        sample = np.column_stack((sample, std))

        #Convert to dataframe and distinguish continuous features
        df = pd.DataFrame(sample, columns=head[0])
        df_continuous_features = df.drop('STD', axis=1).columns.tolist()

        #Data and model object for DiCE
        data = dice_ml_custom.Data(dataframe=df, continuous_features=df_continuous_features, outcome_name='STD')
        dice_model = dice_ml_custom.Model(model=NNmodel, backend='PYT', model_type='regressor')
        exp_random = dice_ml_custom.Dice(data, dice_model, method='random')



        #Generate counterfactual explanations
        cf = exp_random.generate_counterfactuals(df.drop('STD', axis=1), 
                                                verbose=False, 
                                                total_CFs= 1, 
                                                desired_range=[0.75, 0.85],
                                                random_seed = 2,
                                                time_series=False,
                                                variance=True)
        
        # cf.visualize_as_dataframe(show_only_changes=True)
        
        cf_total = cf.cf_examples_list[0].final_cfs_df
        
        
        if cf_total is not None:
            #Save cf_result to file
            save_to = os.path.join(project_path, f'DiCE_uncertainty/{BayDet}_cf_results/inputs', DATASET, f'{noisy}')
            if not os.path.exists(save_to): os.makedirs(save_to)
            file_name = os.path.join(save_to, "cf_{0:0=5d}_{1:0=3d}.csv".format(sample_id, label))
            cf_total.to_csv(file_name, index=False)
            # print(f'Saved to: {file_name}')

        else:
            #If no cf found, save a file containing NaN
            save_to = os.path.join(project_path, f'DiCE_uncertainty/{BayDet}_cf_results/inputs', DATASET, f'{noisy}')
            if not os.path.exists(save_to): os.makedirs(save_to)
            file_name = os.path.join(save_to, "cf_{0:0=5d}_{1:0=3d}.csv".format(sample_id, label))
            # no_cf = pd.DataFrame([[np.NAN for _ in range(len(sample[0]))]], columns=head[0])
            no_cf = df
            print('no_cf_found')
            no_cf.to_csv(file_name, index=False)
            # print(f'Saved to: {file_name}')


if __name__ == '__main__':

    start = time.time()

    num_cores = mp.cpu_count() - 1
    num_cores = min(128, num_cores)

    

    # file_paths = file_paths[0:int(sample_len[0][0])+184] #only looking at the first engine
    # file_paths = file_paths[0:170]



    chunks = chunk_list(test_to_cf, min(num_cores, len(test_to_cf)))
    print('Starting multiprocessing')
    print(f'Number of available cores: {num_cores}')
    print(f'Number of samples: {len(test_to_cf)}')


    with mp.Pool(processes=min(num_cores, len(chunks))) as pool:
        list(tqdm.tqdm(pool.imap_unordered(CMAPSS_counterfactuals, chunks), total=len(chunks)))

    # CMAPSS_counterfactuals(chunks[0])

    # p_map(CMAPSS_counterfactuals, file_paths, num_cpus=num_cores, total=len(file_paths), desc= 'Processing')

    end = time.time()
    print('Processing ended')
    print('Time elapsed:', np.round((end-start)/60, 2), 'minutes')



    #%%