#Relevance Vector Machine model

#%% import dependencies
import glob
import sys
import os

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

import numpy as np
import matplotlib.pyplot as plt

from sklearn_rvm import EMRVR
from joblib import dump, load

from variables import *
from Data_loader import CustomDataset

#Load definitions

def getdata(folder_path):
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.txt')]
    file_paths.sort()

    all_samples = []
    all_labels = []

    for file in file_paths:
        data = np.genfromtxt(file, delimiter=" ", dtype=np.float32)
        label = float(file[-7:-4])
        label = np.float32(label)

        #Flatten sample and combine with RUL
        sample = [[element for row in data for element in row]] #flatten time series sample into format [(sensor 1, timestep 0),...(sensor n, timestep w)]

        all_samples.append(sample)
        all_labels.append(label)

    stacked_samples = np.vstack(all_samples)
    stacked_labels = np.array(all_labels, dtype=np.float32)

    return stacked_samples, stacked_labels

current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory

TRAINDATASET = os.path.abspath(os.path.join(project_path, f'data/{DATASET}/min-max/train'))
TESTDATASET = os.path.abspath(os.path.join(project_path, f'data/{DATASET}/min-max/test'))

TRAIN = True
CV = False #Cross validation, if Train = True and CV = False, the model will train on the entire data-set


x_train, y_train = getdata(TRAINDATASET)
x_test, y_test = getdata(TESTDATASET)

#%% Train the model
RVM_model = EMRVR(kernel='rbf', gamma='auto')
RVM_model.fit(x_train[0:20], y_train[0:20])
# dump(RVM_model, 'RVM/RVM_model.joblib')
# %%

# y, y_std = RVM_model.predict(x_test, return_std = True)

