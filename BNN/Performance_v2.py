#%% Evaluating overall performance of models
import os
import glob
import csv
import numpy as np
import pandas as pd

import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import norm

import matplotlib.pyplot as plt

from torch import load
from tqdm import tqdm
import torch
from torchvision.transforms import ToTensor
import time
import sys
import scipy.stats as stats
import json
from itertools import chain
from collections import defaultdict
from prettytable import PrettyTable



# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

from DNN import NeuralNetwork
# from DNN_vizualizer import y_pred_lst, D_RMSE

from variables import *

def s_score(error):
    """S score specifically for RUL prediction. The characteristic of this scoring funciton is that it favours early predictions
    more than late predicitons.

    Args:
        error (float): Predicted RUL - True RUL

    Returns:
        float: scoring function result
    """
    if error < 0:
        return np.exp(-error/13) - 1
    else:
        return np.exp(error/10) - 1
    
def plot_mean_and_percentile(mean, variance, percentile=90, upper_lower = 'upper'):
    std_dev = np.sqrt(variance)

    # Calculate the z-score corresponding to the desired percentile
    z_score = stats.norm.ppf(percentile / 100)

    # Calculate the values corresponding to the mean, lower percentile, and upper percentile
    lower_percentile_value = mean - z_score * std_dev
    upper_percentile_value = mean + z_score * std_dev

    if upper_lower == 'upper':
        return upper_percentile_value
    elif upper_lower == 'lower':
        return lower_percentile_value
    
def alpha_splits(means, vars, trues, key_ranges, alpha=0.2):

    pred_dict = defaultdict(list)
    for key, value in zip(trues, zip(means, vars, trues)):
        pred_dict[key].append(value)

    pred_dict = dict(pred_dict)

    pred_split_dict = defaultdict(list)

    for key, value in pred_dict.items():
        for high, low in key_ranges:
            if low <= key < high:
                pred_split_dict[high, low].append(value)

    pred_split_dict = dict(pred_split_dict)

    alpha_split_dict = defaultdict(list)

    for key in key_ranges:
        alphas = []
        if key in pred_split_dict:
            for pred in pred_split_dict[key]:
                for sub_pred in pred:
                    mean, var, true = sub_pred
                    # upper = plot_mean_and_percentile(mean, var, upper_lower='upper')
                    # lower = plot_mean_and_percentile(mean, var, upper_lower='lower')
                    # alpha = max(np.abs(true - upper), np.abs(true - lower))/true if int(true) != 0 else 0
                    # alphas.append(np.round(alpha,2))
                    alphas.append(alpha_dist(upper_bound=true*(1+alpha), lower_bound=true*(1-alpha), mean=mean, stdev=np.sqrt(var)))
            alpha_split_dict[str(key)] = np.mean(alphas)
        else:
            alpha_split_dict[str(key)] = np.NaN

    alpha_split_dict = dict(alpha_split_dict)

    return alpha_split_dict
    
def RMSE_split(errors, trues, key_ranges):

    # Create a dict for the errors
    error_dict = defaultdict(list)
    for key, value in zip(trues, errors):
        error_dict[key].append(value)

    # Convert the defaultdict to a regular dictionary
    error_dict = dict(error_dict)

    # Create a dict that splits the error values over certain RUL sections
    error_split_dict = defaultdict(list)


    # Split up the errors according to their true RUL
    for key, value in error_dict.items():
        for high, low in key_ranges:
            if low <= key < high:
                error_split_dict[high, low].append(value)

    # Convert to dict split up into sections
    error_split_dict = dict(error_split_dict)

    RMSE_splits = defaultdict(list)
    for key in key_ranges:
        if key in error_split_dict:
            flat_errors = list(chain(*error_split_dict[key]))
            squared_errors = [error**2 for error in flat_errors]
            RMSE = np.sqrt(np.mean(squared_errors))
            RMSE_splits[str(key)] = np.round(RMSE,2)
        else:
            RMSE_splits[str(key)] = np.NaN

    return dict(RMSE_splits)

def var_split(vars, trues, key_ranges):
    # Create a dict for the variances
    var_dict = defaultdict(list)
    for key, value in zip(trues, vars):
        var_dict[key].append(value)

    # Convert the defaultdict to a regular dictionary
    var_dict = dict(var_dict)

    # Create a dict that splits the variance values over certain RUL sections
    var_split_dict = defaultdict(list)


    # Split up the variance according to their true RUL
    for key, value in var_dict.items():
        for high, low in key_ranges:
            if low <= key < high:
                var_split_dict[high, low].append(value)

    #take the average
    var_splits = defaultdict(list)
    for key in key_ranges:
        if key in var_split_dict:
            flattend_list = [item for sublist in var_split_dict[key] for item in sublist]
            var_splits[str(key)] = np.average(flattend_list)
        else:
            var_splits[str(key)] = np.nan

    return dict(var_splits)

def alpha_dist(lower_bound, upper_bound, mean, stdev):
    """Calculates the percentage of of the distribution that lies within a given range

    Args:
        lower_bound (float): lower bound of range
        upper_bound (float): upper bound of range
        mean (float): mean of distribution
        stdev (float): standard deviation of distribution

    Returns:
        float: Percentage of distribution within specified range
    """
    prob_lower = norm.cdf(lower_bound, loc=mean, scale=stdev)
    prob_upper = norm.cdf(upper_bound, loc=mean, scale=stdev)

    percentage_in_range = (prob_upper - prob_lower)

    return percentage_in_range


test_paths = ['denoised-orig','denoised-NOCF', 'denoised-CF', 'noisy-orig', 'noisy-NOCF', 'noisy-CF']
# test_paths = ['denoised-orig', 'noisy-orig']
key_ranges = [(float('inf'), 120), (120, 60), (60, 30), (30, 10), (10, 0)]
# key_ranges = [(float('inf'), 0)]

total_RMSE_dict = {str(key) : {test_path : [] for test_path in test_paths} for key in key_ranges}
total_var_dict = {str(key) : {test_path : [] for test_path in test_paths} for key in key_ranges}
total_std_dict = {str(key) : {test_path : [] for test_path in test_paths} for key in key_ranges}
total_alpha_dict = {str(key) : {test_path : [] for test_path in test_paths} for key in key_ranges}


for test_path in test_paths:
#%%
    #import BNN results: every file represents 1 engine
    BNN_result_path = os.path.join(project_path, 'BNN/BNN_results', DATASET, test_path)
    engines= glob.glob(os.path.join(BNN_result_path, '*.json'))  # Get a list of all file paths in the folder
    engines.sort() 


    #Gather data for all engines
    BNN_scores = []

    B_means = [] #list of Bayesian prediction means
    B_vars = [] #list of Bayesian prediction variance


    trues = [] #list of true RUl values

    B_RMSE_lst = [] #list of Bayesian RMSE values

    B_errors = [] #list of Bayesian errors

    ave_alphas = []
    total_alphas = []

    # engines = engines[engine_eval:engine_eval+1] if not TEST_SET else engines #only evaluate a single engine

    for engine in engines:
        engine_id = int(engine[-8:-5])
        with open(engine, 'r') as jsonfile:
            results = json.load(jsonfile)
        
        #BNN results
        mean_pred_lst = results['mean'] #Mean of the RUL predictions over engine life
        var_pred_lst = results['var'] #Variance of the RUL predictions over engine life
        true_lst = results['true'] #Ground truth RUL over engine life
        

            
        #%% Get error data
        BNN_error = [(mean_pred_lst[i] - true_lst[i]) for i in range(len(true_lst))] #BNN error
        B_RMSE = np.round(np.sqrt(np.mean([(mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))])), 2) #Root Mean Squared error of Bayesian prediciton

        B_RMSE_lst.append(B_RMSE)


        #Calculate scores
        BNN_score = sum([s_score(i) for i in BNN_error])

        BNN_scores.append(BNN_score)

        B_means.append(mean_pred_lst)
        B_vars.append(var_pred_lst)

        trues.append(true_lst)

        B_errors.append(BNN_error)

        #calculate alphas
        alpha = 0.2
        alphas = [alpha_dist(lower_bound=true_lst[i]*(1-alpha), upper_bound=true_lst[i]*(1+alpha), mean=mean_pred_lst[i], stdev=np.sqrt(var_pred_lst[i])) for i in range(len(mean_pred_lst))]
        ave_alphas.append(np.mean(alphas))
        total_alphas.append(alphas)


    print(f'BNN score: {sum(BNN_scores)}')



    total_B_RMSE_splits = {str(key) : [] for key in key_ranges}
    total_B_var_splits = {str(key) : [] for key in key_ranges}
    total_alpha_splits = {str(key) : [] for key in key_ranges}

    for B_mean, B_var, B_error, true in zip(B_means, B_vars, B_errors, trues ):
        B_RMSE_splits = RMSE_split(B_error, true, key_ranges)

        B_var_splits = var_split(B_var, true, key_ranges)

        B_alpha_splits = alpha_splits(B_mean, B_var, true, key_ranges)

        for key in key_ranges:
            key = str(key)
            if not np.isnan(B_RMSE_splits[key]):
                total_B_RMSE_splits[key].append(B_RMSE_splits[key])
                total_RMSE_dict[key][test_path].append(B_RMSE_splits[key])

            if not np.isnan(B_var_splits[key]):
                total_B_var_splits[key].append(B_var_splits[key])
                total_var_dict[key][test_path].append(B_var_splits[key])
                total_std_dict[key][test_path].append(np.sqrt(B_var_splits[key]))

            if not np.isnan(B_alpha_splits[key]):
                total_alpha_splits[key].append(B_alpha_splits[key])
                total_alpha_dict[key][test_path].append(B_alpha_splits[key])



    ave_B_RMSE_splits = [np.mean(total_B_RMSE_splits[key]) for key in total_B_RMSE_splits.keys()]

    ave_B_var_splits = [np.mean(total_B_var_splits[key]) for key in total_B_var_splits.keys()]

    ave_alpha_splits = [np.mean(total_alpha_splits[key]) for key in total_alpha_splits.keys()]

    tab = PrettyTable(key_ranges)
    tab.add_row(np.round(ave_B_RMSE_splits, 2))
    tab.add_row(np.round(ave_B_var_splits, 2))
    tab.add_row(np.round(ave_alpha_splits, 2))
    # tab.add_row(np.round(ave_alpha_splits, 2), divider=True)
    tab.add_column('Overall average', [np.round(np.mean(B_RMSE_lst),2), 
                            np.round(np.mean([np.mean(B_vars[i]) for i in range(len(B_vars))]),2),
                            np.round(np.mean(ave_alphas),2)]
                            )
    tab.add_column('Metric', ['Average RMSE', 
                            'Average Variance',
                            'Average distribution in alpha'], align='r')
    print('RMSE (cycles) for RUL sections')
    print(tab)

    print(f'Average percentage of {test_path} distribution in alpha range: {np.mean(ave_alphas)}')


#plot results
# Create subplots
fig, axs = plt.subplots(3, len(key_ranges), figsize=(16, 12))
# fig.suptitle('Overal model performance per RUL section')

for i, key in enumerate(key_ranges):
    key = str(key)
    # for j, test_path in enumerate(test_paths):
    positions = np.arange(0, len(test_paths))
    # Create Box plots and add them to subplots
    axs[0,i].boxplot(list(total_RMSE_dict[key].values()), labels=total_RMSE_dict[key].keys(), positions=positions)
    axs[0,i].set_ylim(bottom=0, top=35)
    axs[0,i].grid(visible=True, which='both', axis='both', alpha=0.5)
    axs[0,i].tick_params(axis='x', labelrotation = 45)
    axs[0,0].set_ylabel('RMSE [cycles]')

    # Scatter plots for original data points
    for box_key, box_value in total_RMSE_dict[key].items():
        axs[0,i].scatter(np.repeat(box_key, len(box_value)), box_value, alpha=0.7, color='orange', s=15)

    axs[1,i].boxplot(total_std_dict[key].values(), labels=total_std_dict[key].keys(), positions=positions)
    axs[1,i].set_ylim(bottom=0, top=14)
    axs[1,i].grid(visible=True, which='both', axis='both', alpha=0.5)
    axs[1,i].tick_params(axis='x', labelrotation = 45)
    axs[1,0].set_ylabel('STD [cycles]')

    # Scatter plots for original data points
    for box_key, box_value in total_std_dict[key].items():
        axs[1,i].scatter(np.repeat(box_key, len(box_value)), box_value, alpha=0.7, color='orange', s=15)

    axs[2,i].boxplot(total_alpha_dict[key].values(), labels=total_alpha_dict[key].keys(), positions=positions)
    axs[2,i].set_ylim(bottom=0, top=1.0)
    axs[2,i].grid(visible=True, which='both', axis='both', alpha=0.5)
    axs[2,i].tick_params(axis='x', labelrotation = 45)
    axs[2,0].set_ylabel(f'Distribution within alpha={alpha}')

    # Scatter plots for original data points
    for box_key, box_value in total_alpha_dict[key].items():
        axs[2,i].scatter(np.repeat(box_key, len(box_value)), box_value, alpha=0.7, color='orange', s=15)

    axs[0,i].set_title(f'RUL section: {key}')


plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout
plt.savefig(f'/Users/jillesandringa/Documents/AE/MSc/Thesis/Graphs_and_figures/overall_performance_sections.pdf', format='pdf', bbox_inches='tight')
plt.show()