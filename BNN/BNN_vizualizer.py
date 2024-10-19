#%%Vizualizing script for the BNN model including DNN and uncertainty predictions
import os
import glob
import csv
import numpy as np

import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from torch import load
from tqdm import tqdm
import torch
from torchvision.transforms import ToTensor
import time
import sys
import scipy.stats as stats
import json

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

from DNN import NeuralNetwork
# from DNN_vizualizer import y_pred_lst, D_RMSE

from variables import *

#definitions
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

def calculate_alpha(means, std_devs, true_values, confidence_level=0.9):
    alphas = []

    for mean, std_dev, true_value in zip(means, std_devs, true_values):
        # Calculate the z-score corresponding to the desired confidence level
        z_score = stats.norm.ppf(0.5 + confidence_level / 2)

        # Calculate the distance from the mean such that at least 90% of the distribution lies within
        distance = z_score * std_dev

        # Calculate the lower and upper bounds of the interval
        lower_bound = mean - distance
        upper_bound = mean + distance

        # Calculate the distance from the true value
        alpha = max(np.abs(upper_bound - true_value)/true_value*100, np.abs(true_value - lower_bound)/true_value*100) if true_value != 0 else 0
        alphas.append(alpha)

        # distance_from_true_value = np.abs(true_value - upper_bound) if true_value > mean else np.abs(true_value - lower_bound)
        # distance_from_true_value = distance_from_true_value/true_value * 100 if true_value != 0 else distance_from_true_value

        # distances.append(distance_from_true_value)

    return max(alphas)

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

def alpha_det(lower_bound, upper_bound, pred):
    """Alpha lambda performance over time

    Args:
        lower_bound (float): lower bound of range
        upper_bound (float): upper bound of range
        pred (float): Deterministic prediction of RUL

    Returns:
        int: 1 if prediction within bounds, 0 otherwise
    """

    if pred <= upper_bound and pred >= lower_bound:
        return 1
    else:
        return 0
    
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

start = time.time()

show_cf = False
GIF = False
alpha = 0.2 #set the alpha bounds
engine_eval = 3

NOISY = False
CF = False
NOCF = False
INCREASE = False

noisy = 'noisy' if NOISY else 'denoised'
cf = 'CF' if CF else ('NOCF' if NOCF else 'orig')
increase = 'increase' if INCREASE else 'decrease'
#%%
#import BNN results: every file represents 1 engine
BNN_result_path = os.path.join(project_path, 'BNN/BNN_results', DATASET, f'{noisy}-{cf}')
engines= glob.glob(os.path.join(BNN_result_path, '*.json'))  # Get a list of all file paths in the folder
engines.sort() 

#import CF results: every file represents 1 engine
CF_result_path = os.path.join(project_path, 'DiCE/BNN_cf_results/outputs', DATASET, increase, noisy)
CF_engines= glob.glob(os.path.join(CF_result_path, '*.json'))  # Get a list of all file paths in the folder
CF_engines.sort() 

#import DNN results: every file represents 1 engine
DNN_result_path = os.path.join(project_path, 'BNN/DNN_results', DATASET, noisy)
DNN_engines= glob.glob(os.path.join(DNN_result_path, '*.json'))  # Get a list of all file paths in the folder
DNN_engines.sort() 

for engine in engines[engine_eval: engine_eval+1]:
    engine_id = int(engine[-8:-5])
    with open(engine, 'r') as jsonfile:
        results = json.load(jsonfile)
    
    #BNN results
    mean_pred_lst = results['mean'] #Mean of the RUL predictions over engine life
    var_pred_lst = results['var'] #Variance of the RUL predictions over engine life
    true_lst = results['true'] #Ground truth RUL over engine life

    with open(CF_engines[engine_id], 'r') as jsonfile:
        CF_results = json.load(jsonfile)
    
    #Counterfactual results
    CF_mean_pred_lst = CF_results['mean'] #Mean of the RUL predictions over engine life
    CF_var_pred_lst = CF_results['var'] #Variance of the RUL predictions over engine life
    CF_true_lst = CF_results['true'] #Ground truth RUL over engine life

    with open(DNN_engines[engine_id], 'r') as jsonfile:
        DNN_results = json.load(jsonfile)
    
    #Deterministic results
    y_pred_lst = DNN_results['pred'] #RUl predictions over engine life

        
    #%% Plot data
    BNN_error = [(mean_pred_lst[i] - true_lst[i]) for i in range(len(true_lst))] #BNN error
    DNN_error = [(y_pred_lst[i] - true_lst[i]) for i in range(len(true_lst))]#DNN error
    B_RMSE = np.round(np.sqrt(np.mean([(mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))])), 2) #Root Mean Squared error of Bayesian prediciton
    D_RMSE = np.round(np.sqrt(np.mean([(y_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))] )), 2) #Root Mean Squared error of Deterministic prediciton

    x_plot = np.arange(len(mean_pred_lst))

    # Create a subplot with 2 rows and 2 column for the sub-plots
    fig = make_subplots(rows=2, cols=2, shared_xaxes=False, vertical_spacing=0.1,
                        subplot_titles=[f'RUL prediction of engine {engine_id}', f'Distribution within \u03B1 +-{alpha*100}%', 'Prediction distribution'])

    #main figure containing data independent of selector position
    main_figure = [
                    go.Scatter(x=x_plot, 
                                y=mean_pred_lst, 
                                mode='lines', 
                                line=dict(color='blue'),
                                visible=False,
                                name=f'Bayesian Mean Predicted RUL values for engine {engine_id}, RMSE = {B_RMSE}'),
                    go.Scatter(x=x_plot,
                                y=y_pred_lst,
                                name= f'Deterministic Predicted RUL values, RMSE = {D_RMSE}',
                                mode='lines',
                                visible=False,
                                line=dict(color='orange')),
                    go.Scatter(x=x_plot,
                                y=true_lst,
                                name='True RUL values',
                                mode='lines',
                                visible=False,
                                line=dict(color='red')),
                    go.Scatter(x=np.concatenate((x_plot, x_plot[::-1])), 
                                y=np.concatenate((np.array([plot_mean_and_percentile(mean_pred_lst[i], var_pred_lst[i], percentile=90, upper_lower='upper') for i in range(len(x_plot))]), 
                                                  np.array([plot_mean_and_percentile(mean_pred_lst[i], var_pred_lst[i], percentile=90, upper_lower='lower') for i in range(len(x_plot))])[::-1])),
                                fill='toself',  # Fill to next y values
                                fillcolor='rgba(0, 100, 80, 0.2)',  # Color of the filled area
                                line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
                                visible=False,
                                name='90th percentile',
                                hoverinfo='skip'),
                     go.Scatter(x=x_plot, 
                                y=CF_mean_pred_lst, 
                                mode='lines', 
                                line=dict(color='green'),
                                visible=False,
                                name=f'Counterfactual Mean Predicted RUL values for engine {engine_id}'),    
                    go.Scatter(x=np.concatenate((x_plot, x_plot[::-1])), 
                                y=np.concatenate((np.array([plot_mean_and_percentile(CF_mean_pred_lst[i], CF_var_pred_lst[i], percentile=90, upper_lower='upper') for i in range(len(x_plot))]), 
                                                  np.array([plot_mean_and_percentile(CF_mean_pred_lst[i], CF_var_pred_lst[i], percentile=90, upper_lower='lower') for i in range(len(x_plot))])[::-1])),
                                fill='toself',  # Fill to next y values
                                fillcolor='rgba(0, 100, 80, 0.2)',  # Color of the filled area
                                line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
                                visible=False,
                                name='90th percentile',
                                hoverinfo='skip'),        
                    go.Scatter(x=np.concatenate((x_plot, x_plot[::-1])), 
                                y=np.concatenate((np.array([i*(1.0+alpha) for i in true_lst]), np.array([i*(1.0-alpha) for i in true_lst])[::-1])),
                                fill='toself',  # Fill to next y values
                                fillcolor='rgba(0, 80, 200, 0.15)',  # Color of the filled area
                                visible=False,
                                line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
                                name=f'\u03B1 +-{alpha*100}%, \u03BB',
                                hoverinfo='skip')
                ]
                

    std_dev = np.sqrt(var_pred_lst)
    CF_std_dev = np.sqrt(CF_var_pred_lst)
    for i in range(len(x_plot)):
        for j in range(len(main_figure)):
            fig.add_trace(main_figure[j], row=1, col=1)
        
        #Plot data dependent on selector position

        #x,y data for Bayesian probability distribution at selected cycle
        y_sub = np.linspace(mean_pred_lst[i] - 3 * std_dev[i], mean_pred_lst[i] + 3 * std_dev[i], 100)
        x_sub = norm.pdf(y_sub, mean_pred_lst[i], std_dev[i])

        #x,y data for Counterfactual Bayesian probability distribution at selected cycle
        CF_y_sub = np.linspace(CF_mean_pred_lst[i] - 3 * CF_std_dev[i], CF_mean_pred_lst[i] + 3 * CF_std_dev[i], 100)
        CF_x_sub = norm.pdf(CF_y_sub, CF_mean_pred_lst[i], CF_std_dev[i])

        fig.add_trace(go.Scatter(x=x_sub, 
                                y=y_sub, 
                                visible=False,
                                mode='lines', 
                                line=dict(color='rgba(0, 20, 100, 0.2)'),
                                fill='tozeroy',
                                name='RUL prediction distribution'),
                                row=2,
                                col=1)
        
        fig.add_trace(go.Scatter(x=CF_x_sub, 
                                y=CF_y_sub, 
                                visible=False,
                                mode='lines', 
                                line=dict(color='rgba(0, 100, 20, 0.2)'),
                                fill='tozeroy',
                                name=' Counterfactual RUL prediction distribution'),
                                row=2,
                                col=1)
        
        fig.add_trace(go.Scatter(x=np.array([0, max(x_sub)]),
                                y=np.array([true_lst[i], true_lst[i]]),
                                visible=False,
                                mode='lines',
                                line=dict(dash='dash', color='blue'),
                                name='True RUL'),
                                row=2,
                                col=1)
        
        fig.add_trace(go.Scatter(x=np.array([0, max(x_sub)]),
                                y=np.array([y_pred_lst[i], y_pred_lst[i]]),
                                visible=False,
                                mode='lines',
                                line=dict(dash='dash', color='orange'),
                                name='Deterministic Predicted RUL'),
                                row=2,
                                col=1)
        
        fig.add_trace(go.Scatter(x=np.array([i, i]),
                                y=np.array([0, mean_pred_lst[i]]),
                                visible=False,
                                mode='lines',
                                line=dict(dash='dash', color='black'),
                                showlegend=False,
                                name='Cycle indicator'),
                                row=1,
                                col=1)
        
        fig.add_trace(go.Scatter(x=np.array([i, i]),
                                y=np.array([0, 1]),
                                visible=False,
                                mode='lines',
                                line=dict(dash='dash', color='black'),
                                showlegend=False,
                                name='Cycle indicator'),
                                row=1,
                                col=2)
         
        fig.add_trace(go.Scatter(x=x_plot,
                                y = np.array([alpha_dist(true_lst[i]*(1-alpha), true_lst[i]*(1+alpha), mean_pred_lst[i], np.sqrt(var_pred_lst[i])) for i in range(len(x_plot))]),
                                visible=False,
                                mode='lines',
                                fill='tozerox',
                                line=dict(color='rgba(0, 80, 200, 0.5)'),
                                name=f'Prediction within \u03B1 +-{alpha*100}%'),
                                row=1,
                                col=2)
        
        # fig.add_trace(go.Scatter(x=x_plot,
        #                         y=np.array([calculate_alpha(mean_pred_lst, np.sqrt(var_pred_lst), true_lst, confidence_level=0.9) for i in range(len(x_plot))]),
        #                         visible=False,
        #                         mode='lines',
        #                         line=dict(color='orange'),
        #                         name=f'Prognostic horizon: Distance from true values where 90% of predictions lie'),
        #                         row=1,
        #                         col=2)
        
        fig.add_trace(go.Scatter(x=x_plot,
                                 y=np.array([s_score(i) for i in BNN_error]),
                                 visible=False,
                                 mode='lines',
                                 line=dict(color='rgba(0, 80, 200, 0.5)'),
                                 name='BNN Scoring function (prefers early predictions than late predictions)'),
                                 row=2,
                                 col=2
                                 )
        fig.add_trace(go.Scatter(x=x_plot,
                                 y=np.array([s_score(i) for i in DNN_error]),
                                 visible=False,
                                 mode='lines',
                                 line=dict(color='orange'),
                                 name='DNN Scoring function (prefers early predictions than late predictions)'),
                                 row=2,
                                 col=2
                                 )

        #Invisible x axis line to keep axis consistent
        fig.add_trace(go.Scatter(x=np.array([0, 0.55]),
                                y=np.array([0, 0]),
                                visible=False,
                                mode='lines',
                                line=dict(color='rgba(255, 255, 255, 0)'),
                                showlegend=False),
                                row=2,
                                col=1)


    n_traces = int(len(fig.data)/(max(x_plot)+1))
    for i in range(n_traces):
        fig.data[i].visible = True

    #Logic for slider: slider will select cycle to display
    steps = []
    for i in range(len(x_plot)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Evaluated cycle: " + str(i)}],  # layout attribute
        )
        for j in range(i*n_traces, (i+1)*n_traces):
            step["args"][0]["visible"][j] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Cycle: "},
        pad={"t": 50},
        steps=steps
    )]


    fig.update_layout(
        sliders=sliders,
        title=f'RUL prediction of engine {engine_id}', 
        showlegend=True,

    )

    fig.update_xaxes(title_text='Cycles', row=1, col=1)
    fig.update_yaxes(title_text='RUL [cycles]', row=1, col=1)

    fig.update_xaxes(title_text='Probability density', row=2, col=1)
    fig.update_yaxes(title_text='RUL [cycles]', row=2, col=1)

    fig.update_xaxes(title_text='Cycles', row=1, col=2)
    fig.update_yaxes(title_text='Distance from true values [%]', row=1, col=2)

    fig.update_xaxes(title_text='Cycles', row=2, col=2)
    fig.update_yaxes(title_text='Score [-]', row=2, col=2)

    fig.show()

    # Export the figure to an HTML file
    pyo.plot(fig, filename='interactive-plots/prediction_fig.html', auto_open=False)

    if GIF == True:
        # Create an empty figure
        animation_fig = go.Figure()

        # Capture frames of the plot at each step
        frames = []
        loop = tqdm(x_plot)
        for i, _ in enumerate(x_plot):
            for j in range(n_traces):
                fig.data[i * n_traces + j].visible = True
            frame = go.Frame(data=fig.data, name=f'Frame {i}')
            frames.append(frame)
            for j in range(n_traces):
                fig.data[i * n_traces + j].visible = False

        # Update the empty figure with the captured frames
        animation_fig.frames = frames

        # Define the layout for the animation
        animation_fig.update_layout(
            sliders=sliders,
            title=f'RUL prediction of engine {engine_id}',
            showlegend=True,
        )

        # Function to update the slider
        def update_slider(frame):
            step = frame % len(x_plot)
            animation_fig.update(frames=[go.Frame(data=fig.data, name=f'Frame {step}')])

        # Create the animation
        ani = FuncAnimation(plt.gcf(), update_slider, frames=len(x_plot) * n_traces)

        # Save the animation as a GIF
        ani.save('interactive-plots/slider_animation.gif', writer='pillow', fps=10)  # Adjust the file name and FPS as needed


# %%
