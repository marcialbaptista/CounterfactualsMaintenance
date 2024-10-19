# -*- coding: utf-8 -*-
# code based on repository by kkangshen
"""C-MAPSS preprocessing."""

#%% import packages
import os
import zipfile

import sys
# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__))))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

from variables import *
import glob
import csv

import numpy as np
np.random.seed(seed=42)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import savgol_filter
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#%% import files
TRAINDATASET = f'data/{DATASET}/min-max/noisy/train'
TESTDATASET = f'data/{DATASET}/min-max/noisy/test'

PLOT = True
GENERATE = False
noisy = True



#%%
def build_train_data(df, out_path, window=30, normalization="min-max", maxRUL=120):
    """Build train data.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    out_path : str
        Output path.
    window : int, optional
        Sliding window size.
    normalization : str, optional
        Normalization strategy. Either 'min-max' or 'z-score'.

    Returns
    -------
    MinMaxScaler or StandardScaler
        Scaler used to normalize the data.
    """
    assert normalization in ["z-score", "min-max", "min-max/noisy", "min-max/denoised"], "'normalization' must be either 'z-score' or 'min-max', got '" + normalization + "'."

    # normalize data
    if normalization == "z-score":
        scaler = StandardScaler()
        # df.iloc[:, 1 : df.shape[1]] = scaler.fit_transform(df.iloc[:, 1 : df.shape[1]])
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # df.iloc[:, 1 : df.shape[1]] = scaler.fit_transform(df.iloc[:, 1 : df.shape[1]])
        

    # group by trajectory
    grouped = df.groupby("trajectory_id")

    # compute trajectory min and max length
    min_length = min(len(traj) for traj_id, traj in grouped)
    max_length = max(len(traj) for traj_id, traj in grouped)
    

    # count total number of samples
    total_samples = 0
    for traj_id, traj in grouped:
        t = traj.drop(["trajectory_id"], axis=1).values
        total_samples += len(t) - window + 1
            


    # print info
    print("Number of trajectories = " + str(len(grouped)))
    print("Trajectory min length = " + str(min_length))
    print("Trajectory max length = " + str(max_length))
    print("Number of features = " + str(len(df.columns) - 1))
    print("Number of samples = " + str(total_samples))

    assert window <= min_length, "'window' cannot be greater than " + str(min(len(traj) for traj_id, traj in grouped)) + ", got %d." % window

    # sample each trajectory through sliding window segmentation
    sample_id = 0
    traj_len_lst = []
    flag_1 = False
    flag_2 = False
    test_train = 0.5 #fraction of engines to be used for training, rest will be used for testing
    test_to_cf = 0.4 #fraction of engines to be used for counterfactuals, the rest will be used for evaluation


    #Denoising per trajectory
    denoised_trajectories = []
    for traj_id, traj in grouped:
        if traj_id <= int(test_train*len(grouped)):
            name = "train"
        elif traj_id < int((test_train+test_to_cf)*len(grouped))+1 and not flag_1:
            sample_id = 0
            traj_len_lst = []
            name = "test"
            flag_1 = True
        elif traj_id >= int((test_train+test_to_cf)*len(grouped))+1 and not flag_2:
            sample_id = 0
            traj_len_lst = []
            name = "test_eval"
            flag_2 = True

        t = traj.drop(["trajectory_id"], axis=1).values
        # t = traj.values
        if not noisy:
            for i in range(t.shape[1]):
                t[:,i] = savgol_filter(t[:,i], t.shape[0], 3)  #denoising

        t[:, 0 : t.shape[1]] = scaler.fit_transform(t[:, 0 : t.shape[1]]) #normalization

    #     denoised_trajectories.append(t)

    # #Normalise over the entire data set to create a global normalisation
    # denoised_data =np.concatenate(denoised_trajectories)
    # denoised_data = pd.DataFrame(denoised_data, index=None)
    # denoised_data[denoised_data.columns[1:]] = scaler.fit_transform(denoised_data[denoised_data.columns[1:]])

    # #Split back up into the trajectories to create individual sliding windows
    # grouped = denoised_data.groupby(0)
    # for traj_id, traj in grouped:
    #     while traj_id <= int(test_train*len(grouped)):
    #         name = "train"
    #         break
    #     else:
    #         if not flag:
    #             sample_id = 0
    #             traj_len_lst = []
    #             name = "test"
    #             flag = True

        # t = traj.drop(0, axis=1).values

        num_samples = len(t) - window + 1
        traj_len_lst.append(num_samples)
        for i in range(num_samples):
            sample = t[i : (i + window)]
            # sample = scaler.fit_transform(sample)
            label = min(len(t) - i - window, maxRUL)
            path = os.path.join(out_path, name)
            if not os.path.exists(path): os.makedirs(path)
            file_name = os.path.join(path, "{0:0}_{1:0=5d}-{2:0=3d}.txt".format(name, sample_id, label))
            sample_id += 1
            np.savetxt(file_name, sample, fmt="%.10f")
    
        np.savetxt(os.path.join(path, '0-Number_of_samples.csv'), traj_len_lst, delimiter=",", fmt='% s')
    print("Done.")
    return scaler


def build_validation_data(df, out_path, scaler, window=30, maxRUL=120):
    """Build validation data.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    out_path : str
        Output path.
    scaler : MinMaxScaler or StandardScaler
        Scaler to use to normalize the data.
    window : int, optional
        Sliding window size.
    """
    assert scaler, "'scaler' type cannot be None."

    # group by trajectory
    grouped = df.groupby("trajectory_id")

    # compute trajectory min and max length
    min_length = min(len(traj) for traj_id, traj in grouped)
    max_length = max(len(traj) for traj_id, traj in grouped)

    # count total number of samples
    total_samples = 0
    for traj_id, traj in grouped:
        t = traj.drop(["trajectory_id"], axis=1).values
        total_samples += len(t) - window + 1

    # print info
    print("Number of trajectories = " + str(len(grouped)))
    print("Trajectory min length = " + str(min_length))
    print("Trajectory max length = " + str(max_length))
    print("Number of features = " + str(len(df.columns) - 1))
    print("Number of samples = " + str(total_samples))

    assert window <= min_length, "'window' cannot be greater than " + str(min(len(traj) for traj_id, traj in grouped)) + ", got %d." % window

    # sample each trajectory through sliding window segmentation
    sample_id = 0
    traj_len_lst = []
    #Denoising per trajectory
    denoised_trajectories = []
    for traj_id, traj in grouped:

        # t = traj.drop(["trajectory_id"], axis=1).values
        t = traj.values
        if not noisy:
            for i in range(1,t.shape[1]):
                t[:,i] = savgol_filter(t[:,i], t.shape[0], 3)  #denoising

        denoised_trajectories.append(t)

    #Normalise over the entire data set to create a global normalisation
    denoised_data =np.concatenate(denoised_trajectories)
    denoised_data = pd.DataFrame(denoised_data, index=None)
    denoised_data[denoised_data.columns[1:]] = scaler.fit_transform(denoised_data[denoised_data.columns[1:]])

    #Split back up into the trajectories to create individual sliding windows
    grouped = denoised_data.groupby(0)
    for traj_id, traj in grouped:

        t = traj.drop(0, axis=1).values

        num_samples = len(t) - window + 1
        traj_len_lst.append(num_samples)
        for i in range(num_samples):
            sample = t[i : (i + window)]
            # sample = scaler.fit_transform(sample)
            label = min(len(t) - i - window, maxRUL)          
            path = os.path.join(out_path, "validation")
            if not os.path.exists(path): os.makedirs(path)
            file_name = os.path.join(path, "validation_{0:0=5d}-{1:0=3d}.txt".format(sample_id, label))
            sample_id += 1
            np.savetxt(file_name, sample, fmt="%.10f")

    np.savetxt(os.path.join(path, '0-Number_of_samples.csv'), traj_len_lst, delimiter=",", fmt='% s')
    print("Done.")


def build_test_data(df, file_rul, out_path, scaler, window=30, keep_all=False, maxRUL=120):
    """Build test data.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    file_rul : str
        RUL labels file.
    out_path : str
        Output path.
    scaler : MinMaxScaler or StandardScaler
        Scaler to use to normalize the data.
    window : int, optional
        Sliding window size.
    keep_all : bool, optional
        True to keep all the segments extracted from the series, False to keep only the last one.
    """
    assert scaler, "'scaler' type cannot be None."
    df.iloc[:, 1 : df.shape[1]] = scaler.fit_transform(df.iloc[:, 1 : df.shape[1]])

    # group by trajectory
    grouped = df.groupby("trajectory_id")

    # compute trajectory min and max length
    min_length = min(len(traj) for traj_id, traj in grouped)
    max_length = max(len(traj) for traj_id, traj in grouped)

    # count total number of samples
    total_samples = 0
    if keep_all:
        for traj_id, traj in grouped:
            t = traj.drop(["trajectory_id"], axis=1).values
            total_samples += len(t) - window + 1
    else:
        total_samples = len(grouped)

    # print info
    print("Number of trajectories = " + str(len(grouped)))
    print("Trajectory min length = " + str(min_length))
    print("Trajectory max length = " + str(max_length))
    print("Number of features = " + str(len(df.columns) - 1))
    print("Number of samples = " + str(total_samples))

    assert window <= min_length, "'window' cannot be greater than " + str(min(len(traj) for traj_id, traj in grouped)) + ", got %d." % window
    
    # get ground truth
    rul = np.asarray(file_rul.readlines(), dtype=np.int32).clip(max = maxRUl)   

    # sample each trajectory through sliding window segmentation
    sample_id = 0
    traj_len_lst = []
    if not keep_all:
        #Denoising per trajectory
        denoised_trajectories = []
        for traj_id, traj in grouped:

            # t = traj.drop(["trajectory_id"], axis=1).values
            t = traj.values
            if not noisy:
                for i in range(1,t.shape[1]):
                    t[:,i] = savgol_filter(t[:,i], t.shape[0], 3)  #denoising

            denoised_trajectories.append(t)

        #Normalise over the entire data set to create a global normalisation
        denoised_data =np.concatenate(denoised_trajectories)
        denoised_data = pd.DataFrame(denoised_data, index=None)
        denoised_data[denoised_data.columns[1:]] = scaler.fit_transform(denoised_data[denoised_data.columns[1:]])

        #Split back up into the trajectories to create individual sliding windows
        grouped = denoised_data.groupby(0)
        for traj_id, traj in grouped:

            t = traj.drop(0, axis=1).values

            num_samples = 1
            traj_len_lst.append(num_samples)
            sample = t[-window :]
            # sample = scaler.fit_transform(sample)
            label = min(rul[int(traj_id) - 1], maxRUL)
            path = os.path.join(out_path, "test_set")
            if not os.path.exists(path): os.makedirs(path)
            file_name = os.path.join(path, "test_{0:0=5d}-{1:0=3d}.txt".format(sample_id, label))
            sample_id += 1
            np.savetxt(file_name, sample, fmt="%.10f")
    else:
        #Denoising per trajectory
        denoised_trajectories = []
        for traj_id, traj in grouped:

            # t = traj.drop(["trajectory_id"], axis=1).values
            t = traj.values
            if not noisy:
                for i in range(1,t.shape[1]):
                    t[:,i] = savgol_filter(t[:,i], t.shape[0], 3)  #denoising

            denoised_trajectories.append(t)

        #Normalise over the entire data set to create a global normalisation
        denoised_data =np.concatenate(denoised_trajectories)
        denoised_data = pd.DataFrame(denoised_data, index=None)
        denoised_data[denoised_data.columns[1:]] = scaler.fit_transform(denoised_data[denoised_data.columns[1:]])

        #Split back up into the trajectories to create individual sliding windows
        grouped = denoised_data.groupby(0)
        for traj_id, traj in grouped:

            t = traj.drop(0, axis=1).values

            num_samples = len(t) - window + 1
            traj_len_lst.append(num_samples)
            for i in range(num_samples):
                sample = t[i : (i + window)]
                # sample = scaler.fit_transform(sample)
                label = min(len(t) - i - window + rul[traj_id - 1], maxRUL)
                path = os.path.join(out_path, "test_set")   
                if not os.path.exists(path): os.makedirs(path)
                file_name = os.path.join(path, "test_{0:0=5d}-{1:0=3d}.txt".format(sample_id, label))
                sample_id += 1
                np.savetxt(file_name, sample, fmt="%.10f")

    np.savetxt(os.path.join(path, '0-Number_of_samples.csv'), traj_len_lst, delimiter=",", fmt='% s')
    print("Done.")


def extract_dataframes(file_train, file_test, subset="FD001", validation=0.00):
    """Extract train, validation and test dataframe from source file.
    
    Parameters
    ----------
    file_train : str
        Training samples file.
    file_test : str
        Test samples file.
    subset: str, optional
        Subset. Either 'FD001' or 'FD002' or 'FD003' or 'FD004'.
    validation : float, optional
        Ratio of training samples to hold out for validation.
    
    Returns
    -------
    (DataFrame, DataFrame, DataFrame)
        Train dataframe, validation dataframe, test dataframe.
    """
    assert subset in ["FD001", "FD002", "FD003", "FD004"], "'subset' must be either 'FD001' or 'FD002' or 'FD003' or 'FD004', got '" + subset + "'."

    assert 0 <= validation <= 1, "'validation' must be a value within [0, 1], got %.2f" % validation + "."
  
    df = _load_data_from_file(file_train, subset=subset)
    
    # group by trajectory
    grouped = df.groupby("trajectory_id")

    df_train = []
    df_validation = []
    for traj_id, traj in grouped:
        # randomize train/validation splitting
        if np.random.rand() <= (validation + 0.1) and len(df_validation) < round(len(grouped) * validation):
            df_validation.append(traj)
        else:
            df_train.append(traj)

    # print info
    print("Number of training trajectories = " + str(len(df_train)))
    print("Number of validation trajectories = " + str(len(df_validation)))

    df_train = pd.concat(df_train)

    if len(df_validation) > 0:
        df_validation = pd.concat(df_validation)

    df_test = _load_data_from_file(file_test, subset=subset)

    print("Done.")
    return df_train, df_validation, df_test

    
def _load_data_from_file(file, subset="FD001"):
    """Load data from source file into a dataframe.
    
    Parameters
    ----------
    file : str
        Source file.
    subset: str, optional
        Subset. Either 'FD001' or 'FD002' or 'FD003' or 'FD004'.
    
    Returns
    -------
    DataFrame
        Data organized into a dataframe.
    """
    assert subset in ["FD001", "FD002", "FD003", "FD004"], "'subset' must be either 'FD001' or 'FD002' or 'FD003' or 'FD004', got '" + subset + "'."

    n_operational_settings = 3
    n_sensors = 21

    # read csv
    df = pd.read_csv(file, sep=" ", header=None, index_col=False).fillna(method="bfill")
    df = df.dropna(axis="columns", how="all")

    assert df.shape[1] == n_operational_settings + n_sensors + 2, "Expected %d columns, got %d." % (n_operational_settings + n_sensors + 2, df.shape[1])
    
    df.columns = ["trajectory_id", "t"] + ["setting_" + str(i + 1) for i in range(n_operational_settings)] + ["sensor_" + str(i + 1) for i in range(n_sensors)]

    # drop t
    df = df.drop(["t"], axis=1)

    # drop sensors which are useless according to the literature
    to_drop = [1, 5, 6, 10, 16, 18, 19]
    df = df.drop(["sensor_" + str(d) for d in to_drop], axis=1)

    # drop operating_modes
    df = df.drop(["setting_" + str(i + 1) for i in range(n_operational_settings)], axis=1)

    # if subset in ["FD001", "FD003"]:
    #     # drop operating_modes
    #     df = df.drop(["setting_" + str(i + 1) for i in range(n_operational_settings)], axis=1)



    return df


if __name__ == "__main__":
    if GENERATE:
        """Preprocessing."""
        normalization = "min-max/denoised" if not noisy else "min-max/noisy"
        validation = 0.00
        maxRUl = 120

        for subset, window in [("FD001", 30), ("FD002", 20), ("FD003", 30), ("FD004", 15)]:
            print("**** %s ****" % subset)
            print("normalization = " + normalization)
            print("window = " + str(window))
            print("validation = " + str(validation))

            # read file into memory
        
            file_train = open("data_set/train_" + subset + ".txt")
            file_test = open("data_set/test_" + subset + ".txt")
            file_rul = open("data_set/RUL_" + subset + ".txt")

            print("Extracting dataframes...")
            df_train, df_validation, df_test = extract_dataframes(file_train=file_train, file_test=file_test, subset=subset, validation=validation)

            #%% build train data
            print("Preprocessing training data...")
            scaler = build_train_data(df=df_train, out_path="data/" + subset + "/" + normalization, window=window, normalization=normalization, maxRUL=maxRUl)

            #%% build validation data
            if len(df_validation) > 0:
                print("Preprocessing validation data...")
                build_validation_data(df=df_validation, out_path="data/" + subset + "/" + normalization, scaler=scaler, window=window, maxRUL=maxRUl)

            # build test data
            print("Preprocessing test data...")
            build_test_data(df=df_test, file_rul=file_rul, out_path="data/" + subset + "/" + normalization, scaler=scaler, window=window, keep_all=False, maxRUL=maxRUl)

            # save scaler
            print("Saving scaler object to file...")
            scaler_filename = "data/" + "/" + subset + "/" + normalization + "/scaler.sav"
            joblib.dump(scaler, scaler_filename)

            # close files
            file_train.close()
            file_test.close()
            file_rul.close()
            print("Done.")


    #%% _______________Plot sensor data________________
    if PLOT== True:

        # with open(os.path.join(project_path,f'data/{DATASET}/min-max/noisy/test_eval/0-Number_of_samples.csv')) as csvfile:
        #     sample_len = list(csv.reader(csvfile)) #list containing the amount of cf_samples per engine/trajectory


        # #Input files
        # file_paths = glob.glob(os.path.join(project_path, TESTDATASET, '*.txt'))  # Get a list of all file paths in the folder
        # file_paths.sort() 

        # fig, axes = plt.subplots(nrows=2, 
        #                         ncols=7, 
        #                         sharex=True, 
        #                         figsize=(25, 8))

        # #%% Plot input dataframe
        # sensor = 0
        # m = [2,3,4,7,8,9,11,12,13,14,15,17,20,21] #useful sensors
        # engine = 0
        # engine_len = int(sample_len[engine][0]) #TODO: change later to account for engine length
        # # engine_len = 170

        # #Go over every sensor
        # for ax in axes.ravel():

        #     input_total = [] #2D list containing inputs in sliding window form

        #     #Go over engine lifetime
        #     for i, input in enumerate(file_paths[0:engine_len]):

        #         df_input = pd.read_csv(input, sep=' ', header=None)
        #         true_RUL = float(input[-7:-4])

        #         input = df_input[sensor] # input sample for sensor m[sensor] and timestep i
                
        #         #Add inputs to an overall total list spanning engine lifetime
        #         relative_list = [np.NaN for _ in range(engine_len + len(input)-1)]
        #         counter_relative = relative_list.copy()
        #         input_relative = relative_list.copy()
        #         for j in range(len(input)):
        #             #replace NaN values with values in sliding window format
        #             input_relative[j+i] = input[j]
                
        #         input_total.append(input_relative)


        #     #Take the average value of inputs at every time point
        #     input_continuous = np.nanmean(np.array(input_total), axis=0)

        #     ax.plot(np.arange(len(input_continuous)), input_continuous, label='Filtered sensor input')

        #     ax.set_title('Sensor ' + str(m[sensor]))
        #     ax.set_xlabel('Cycles')
        #     # ax.set_ylim(-1,1)
                
        #     sensor += 1

        # plt.show()

        # Load the data
        data = pd.read_csv(f'{project_path}/raw_sensors.csv', delimiter=';')

        # Convert the DataFrame to a long format
        data_long = pd.melt(data.reset_index(), id_vars='index', value_vars=data.columns[1:])
        data_long.columns = ['Cycles', 'Sensor', 'Reading']

        # Adjust the 'Sensor' column to reflect the correct naming
        data_long['Sensor'] = data_long['Sensor'].apply(lambda x: f'Sensor {int(x) - 1}')

        # Set the aesthetic style of the plots
        sns.set(style="whitegrid")

        # Create a FacetGrid, mapping each sensor to a separate plot
        g = sns.FacetGrid(data_long, col='Sensor', col_wrap=5, sharex=False, sharey=False, height=3, aspect=1.5)
        g = g.map(sns.lineplot, 'Cycles', 'Reading', color='blue').set_titles("{col_name}").set_axis_labels("Cycles", "Reading")

        # Adjust the title of each subplot
        for ax, title in zip(g.axes.flat, data_long['Sensor'].unique()):
            ax.set_title(title)

        # Adjust the layout
        plt.tight_layout()

        # Display the plot
        plt.show()