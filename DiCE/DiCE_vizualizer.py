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
import matplotlib
from trend_classifier import Segmenter

NOISY = False

noisy = 'noisy' if NOISY else 'denoised'

#%% import files
TESTDATASET = f'data/{DATASET}/min-max/{noisy}/test_eval'


with open(os.path.join(project_path,f'{TESTDATASET}/0-Number_of_samples.csv')) as csvfile:
    cf_sample_len = list(csv.reader(csvfile)) #list containing the amount of cf_samples per engine/trajectory

#counterfactual input cf_samples
in_decrease = ['increase', 'decrease']

fig, axes = plt.subplots(nrows=1, 
                            ncols=3, 
                            sharex=True, 
                            figsize=(7, 4))

matplotlib.rcParams['font.size'] = 10


for in_de in in_decrease:
    result_path = os.path.join(project_path, 'DiCE/BNN_cf_results/inputs', DATASET, in_de, noisy)
    cf_samples = glob.glob(os.path.join(result_path, '*.csv'))  # Get a list of all file paths in the folder
    cf_samples.sort()

    #Original inputs
    file_paths = glob.glob(os.path.join(project_path, TESTDATASET, '*.txt'))  # Get a list of all file paths in the folder
    file_paths.sort() 

    
    #%% Plot counterfacutal dataframe
    sensor = 0
    m = [2,3,4,7,8,9,11,12,13,14,15,17,20,21] #useful sensors
    sensor_selection = [2, 17, 20]
    engine = 3

    

    engine_len = int(cf_sample_len[engine][0]) #TODO: change later to account for engine length
    engine_len_prev = sum([int(cf_sample_len[engine - i - 1][0]) for i in range(engine)]) if engine != 0 else 0
    # engine_len = 1

    #Go over every sensor
    for ax in axes.ravel():
        cf_total = [] #2D list containing counterfactual inputs in sliding window form
        orig_total = [] #2D list containing original inputs in sliding window form
        sensor_index = m.index(sensor_selection[sensor])

        #Go over engine lifetime
        for i, cf_sample in enumerate(cf_samples[engine_len_prev: engine_len_prev + engine_len]):

            #Counterfactuals
            cf_df = pd.read_csv(cf_sample)
            cf_RUL = cf_df['RUL']
            cf_df = cf_df.drop('RUL', axis=1)
            cf_df = cf_df.values.reshape(30,14)
            cf_df = pd.DataFrame(cf_df)

            counter = cf_df[sensor_index] #Counterfactual input sample for sensor m[sensor] and timestep i

            #Original inputs
            df_orig = pd.read_csv(file_paths[i+engine_len_prev], sep=' ', header=None)
            true_RUL = float(file_paths[i+engine_len_prev][-7:-4])

            orig = df_orig[sensor_index] #Origninal input sample for sensor m[sensor] and timestep i
            
            #Add counterfactuals and original inputs to an overall total list spanning engine lifetime
            relative_list = [np.NaN for _ in range(engine_len + len(counter)-1)]
            counter_relative = relative_list.copy()
            orig_relative = relative_list.copy()
            for j in range(len(counter)):
                #replace NaN values with cf and original values in sliding window format
                orig_relative[j+i] = orig[j]
                counter_relative[j+i] = counter[j]
                # if np.round(counter[j], 5) != np.round(orig[j],5):
                #     counter_relative[j+i] = counter[j]
        
            cf_total.append(counter_relative)
            orig_total.append(orig_relative)
            # diff = [counter_relative[i] - orig_relative[i] for i in range(len(counter_relative))]
            color = 'red' if in_de == 'decrease' else 'green'
            ax.scatter(np.arange(len(counter_relative)), counter_relative, color=color, s=0.5, alpha=0.5)
            ax.plot(np.arange(len(counter_relative)), orig_relative, color='blue')
            # sns.scatterplot(x=np.arange(len(counter_relative)), y=counter_relative, color=color, ax=ax, s=5, alpha=0.5, legend=False)
            # sns.lineplot(x=np.arange(len(counter_relative)), y=orig_relative, color='blue', ax=ax, alpha=0.75, legend=False)


        #Take the average value of inputs at every time point
        cf_average = np.nanmean(np.array(cf_total), axis=0)
        orig_average = np.nanmean(np.array(orig_total), axis=0)

        #Calculate difference between origninal and counterfactual inputs
        difference = cf_average - orig_average
        difference = difference[30:-30]


        # ax.plot(np.arange(30, 30 + len(difference)), difference, label='Relative counterfactual input', alpha=0.3)
        # ax.fill_between(np.arange(30,30+ len(difference)), difference, where=(difference>0), interpolate=True, color=color, alpha=0.3)
        # ax.fill_between(np.arange(30, 30+ len(difference)), difference, where=(difference<0), interpolate=True, color=color, alpha=0.3)

        ax.set_title('Sensor ' + str(m[sensor_index]))
        ax.set_xlabel('Cycles')
        # ax.set_ylim(-1.5,1.5)
            
        sensor += 1

# axes[0,0].set_ylabel('Sensor input difference')
# axes[1,0].set_ylabel('Sensor input difference')

# fig.suptitle(f'Counterfactual explanations: input difference to achieve +- 10-11 extra cycles')
plt.savefig(f'DiCE/cf_inputs_increase_decrease_engine_{engine}_selection.pdf', format='pdf', bbox_inches='tight')
plt.show()

    
    
# %%