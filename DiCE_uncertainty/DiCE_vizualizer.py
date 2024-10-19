#%% import dependencies
import sys
import os

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

from variables import *
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trend_classifier import Segmenter

#definitions
def plot_segments(data, n):
    """Plots trend segments over original data

    Args:
        segments (Segment): Segment object containing information for linear trends
        n (int): Number of cf_samples in a window
    """
    trend = Segmenter(list(np.arange(len(data))), data.to_list(), n=n)
    trend.calculate_segments()
    df = trend.segments.to_dataframe()
    for i in range(len(df)):
        start = df['start'][i]
        stop = df['stop'][i]
        slope = df['slope'][i]
        offset = df['offset'][i]

        x = np.arange(start=start, stop=stop)
        y = x*slope + offset

        ax.plot(x,y)

#%% import files
TRAINDATASET = f'data/{DATASET}/min-max/noisy/train'
TESTDATASET = f'data/{DATASET}/min-max/noisy/test'


with open(os.path.join(project_path,f'data/{DATASET}/min-max/noisy/test/0-Number_of_samples.csv')) as csvfile:
    cf_sample_len = list(csv.reader(csvfile)) #list containing the amount of cf_samples per engine/trajectory

#counterfactual input cf_samples
result_path = os.path.join(project_path, 'DiCE_uncertainty/BNN_cf_results/inputs', DATASET, 'noisy')
cf_samples = glob.glob(os.path.join(result_path, '*.csv'))  # Get a list of all file paths in the folder
cf_samples.sort()

#Original inputs
file_paths = glob.glob(os.path.join(project_path, TESTDATASET, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort() 

fig, axes = plt.subplots(nrows=2, 
                         ncols=7, 
                         sharex=True, 
                         figsize=(25, 8))

#%% Plot counterfacutal dataframe
sensor = 0
m = [2,3,4,7,8,9,11,12,13,14,15,17,20,21] #useful sensors
engine = 0
engine_len = int(cf_sample_len[engine][0]) #TODO: change later to account for engine length
# engine_len = 1

#Go over every sensor
for ax in axes.ravel():
    cf_total = [] #2D list containing counterfactual inputs in sliding window form
    orig_total = [] #2D list containing original inputs in sliding window form

    #Go over engine lifetime
    for i, cf_sample in enumerate(cf_samples[0:engine_len]):

        #Counterfactuals
        cf_df = pd.read_csv(cf_sample)
        # cf_RUL = cf_df['RUL']
        cf_df = cf_df.drop('STD', axis=1)
        cf_df = cf_df.values.reshape(30,14)
        cf_df = pd.DataFrame(cf_df)

        counter = cf_df[sensor] #Counterfactual input sample for sensor m[sensor] and timestep i

        #Original inputs
        df_orig = pd.read_csv(file_paths[i], sep=' ', header=None)
        true_RUL = float(file_paths[i][-7:-4])

        orig = df_orig[sensor] #Origninal input sample for sensor m[sensor] and timestep i
        
        #Add counterfactuals and original inputs to an overall total list spanning engine lifetime
        relative_list = [np.NaN for _ in range(engine_len + len(counter)-1)]
        counter_relative = relative_list.copy()
        orig_relative = relative_list.copy()
        for j in range(len(counter)):
            #replace NaN values with cf and original values in sliding window format
            counter_relative[j+i] = counter[j]
            orig_relative[j+i] = orig[j]
        
        cf_total.append(counter_relative)
        orig_total.append(orig_relative)
        # diff = [counter_relative[i] - orig_relative[i] for i in range(len(counter_relative))]

        ax.scatter(np.arange(len(counter_relative)), counter_relative, color='orange', s=0.5)
        ax.plot(np.arange(len(counter_relative)), orig_relative, color='blue', linewidth=0.1)


    #Take the average value of inputs at every time point
    cf_average = np.nanmean(np.array(cf_total), axis=0)
    orig_average = np.nanmean(np.array(orig_total), axis=0)

    #Calculate difference between origninal and counterfactual inputs
    difference = cf_average - orig_average

    # ax.plot(np.arange(len(difference)), difference, label='Relative counterfactual input')
    # ax.fill_between(np.arange(len(difference)), difference, where=(difference>0), interpolate=True, color='green', alpha=0.5)
    # ax.fill_between(np.arange(len(difference)), difference, where=(difference<0), interpolate=True, color='red', alpha=0.5)

    ax.set_title('Sensor ' + str(m[sensor]))
    ax.set_xlabel('Cycles')
    # ax.set_ylim(-1,1)
        
    sensor += 1

# axes[0,0].set_ylabel('Sensor input difference')
# axes[1,0].set_ylabel('Sensor input difference')

fig.suptitle(f'Counterfactual explanations: input difference to achieve lower uncertainty')
plt.savefig('DiCE_uncertainty/cf_inputs.png')
plt.show()

    
    
# %%