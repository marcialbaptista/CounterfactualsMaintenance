#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from scipy.signal import savgol_filter
from sklearn.cluster import KMeans

#%%_____________________________definitions___________________________________
def add_rul(g, maxRUL=120):
    """Adds Remaining usefull life column to data frames based on final cycle, but limits it to a maximum value (piece wise linear correction)

    Args:
        g (pandas dataframe): dataframe containing cycle, operational and sensor data

    Returns:
        dataframe: dataframe g including final RUL column
    """
    g['RUL'] = max(g['Cycle']) - g['Cycle']
    g['RUL'] = g['RUL'].clip(upper=maxRUL)
    return g

#%%
#____________________________import data files_________________________________
st = time.time()
names = ['Engine', 'Cycle', 'Op1', 'Op2', 'Op3', 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

df1 = pd.read_csv('data_set/train_FD001.txt', sep=' ', header=None)
df2 = pd.read_csv('data_set/train_FD002.txt', sep=' ', header=None)
df3 = pd.read_csv('data_set/train_FD003.txt', sep=' ', header=None)
df4 = pd.read_csv('data_set/train_FD004.txt', sep=' ', header=None)

print('Files imported: ', time.time() - st, 'seconds')

#remove NaN rows and give column names
df_list = [df1, df2, df3, df4]
new_df_list = []

for df in df_list:
    df = df.drop(26, axis=1)
    df = df.drop(27, axis=1)

    #add names and convert to strings
    df.columns = names
    df.columns = df.columns.astype(str)

    #add RUL to data set
    df = df.groupby('Engine', group_keys=False).apply(add_rul)
    new_df_list.append(df)

#df1 and df3 both have 1 operating condition
df1 = new_df_list[0]
df3 = new_df_list[2]

#df2 and df4 both have 6 operating conditions
df2 = new_df_list[1]
df4 = new_df_list[3]

print('Columns edited: ', time.time() - st, 'seconds')

#%%____________________________________denoising signal___________________________________
denoise_df_list = []

for df in new_df_list:
    engines = np.unique(df['Engine'].values)
    df_denoise = df.copy()

    for engine in engines:
        for name_signal in range(21):
            signal = df.loc[df.Engine == engine, str(name_signal + 1)]
            signal_sav = savgol_filter(signal, 15, 3) #apply a Savitzky-Golav filter to the noisy data
            # signal_mav = signal.rolling(10).mean().values #apply mean average filter
            # signal_mav[:10] = signal[:10].values
            df_denoise.loc[df_denoise.Engine == engine, str(name_signal + 1)] = signal_sav
    denoise_df_list.append(df_denoise)

print('Signal denoised: ', time.time() - st, 'seconds')

#Plot original vs denoised signal
raw_signal = df1.loc[df2.Engine == 1, str(2)]
noisy_signal = denoise_df_list[0].loc[df2.Engine == 1, str(2)]

plt.plot(raw_signal)
plt.plot(noisy_signal)
plt.title('Engine 1, Sensor 2 - Noise removal')
plt.xlabel('Operating Cycles')
plt.show()

#%%____________________________data normalization___________________________________
#data needs to be normalized for model to better process data
#also, the different operating conditions need to be normalized for df2 and df4 to be able to compare

normalize_df_list = []

for i in range(len(denoise_df_list)):

    if i == 0 or i == 2:
        for sensor in range(21):
            data = denoise_df_list[i][str(sensor + 1)]
            data = data/max(data)
            denoise_df_list[i][str(sensor + 1)] = data.values

        normalize_df_list.append(denoise_df_list[i])

    if i == 1 or i == 3:
        kmeans_model = KMeans(n_clusters=6, n_init=10)
        est = kmeans_model.fit(denoise_df_list[i][['Op1','Op2','Op3']])
        clusters = est.labels_
        for sensor in range(21):
            for cluster in np.unique(clusters):
                data_cluster = denoise_df_list[i][clusters == cluster][str(sensor + 1)]
                data_cluster = data_cluster/max(data_cluster)
                denoise_df_list[i].loc[clusters == cluster, str(sensor + 1)] = data_cluster.values

        normalize_df_list.append(denoise_df_list[i])

print('Data normalized: ', time.time() - st, 'seconds')

#plot normalized data
df1 = normalize_df_list[3]

fig, axes = plt.subplots(nrows=3, ncols=7, sharex=True,
                                    figsize=(25, 8))
id_equipment = 1
mask_equip1 = df1['Engine'] == id_equipment# Select column Equipment with value x
nrow = 0
ncol = 0

m=1
for ax in axes.ravel():
    signal = df1.loc[mask_equip1,str(m)]
    # signal = df1.loc[mask_equip1, f'Op{m}']
    ax.plot(range(len(signal)), signal)
    ax.set_xlabel('Sensor ' + str(m))
    m += 1

plt.show()

#%%______________________Sensor selection________________________
#From the data, it can be seen that sensors 1, 5, 6, 10, 16, 18 and 19 do not provide any usefull data that could be used to model the RUL prediction
#thus they are removed

final_df_list = []
rem_sensors = ['1','5','6','10','16','18','19']

for df in normalize_df_list:
    df = df.drop(rem_sensors, axis=1)
    final_df_list.append(df)

print('Removed 7 sensors:', time.time() - st, 'seconds')

#plot sensor data data
# df1 = final_df_list[0]

# fig, axes = plt.subplots(nrows=2, ncols=7, sharex=True,
#                                     figsize=(25, 8))
# id_equipment = 1
# mask_equip1 = df1['Engine'] == id_equipment# Select column Equipment with value x
# nrow = 0
# ncol = 0

# i = 0
# m = [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
# for ax in axes.ravel():
#     signal = df1.loc[mask_equip1,str(m[i])]
#     ax.plot(range(len(signal)), signal)
#     ax.set_xlabel('Sensor ' + str(m[i]))
#     i += 1

# plt.show()

# #%%_________________Data export____________________________
# #export the pre processed data files to csv

# for i in range(len(final_df_list)):
#     path = 'processed_data'
#     filepath = os.path.join(path, f'train_FD00{i+1}_processed.csv')
#     final_df_list[i].to_csv(filepath, index=False)

# print('Data exported to csv:', time.time() - st)
# # %%
