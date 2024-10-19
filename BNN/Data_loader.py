import os
import numpy as np
import csv
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, folder_paths):
        self.file_paths = []
        for folder_path in folder_paths:
            file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.txt') or file.endswith('.csv') and not file.endswith('0-Number_of_samples.csv')]
            self.file_paths += file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # Load and process the text file data as per your requirement
        data = self.load_data(file_path)
        label = float(file_path[-7:-4])
        label = np.float32(label)

        # Return the processed data and its corresponding label (if applicable)
        return data, label
    
    def open_csv(self, file_path):
        cf_data = []
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row_number, row in enumerate(csv_reader):
                if row_number == 1:  # Skip the sensor name row
                    cf_data = [np.float32(value) for value in row]

        # Step 2: Remove the final entry
        cf_data = cf_data[:-1]

        # Step 3: Convert the modified second row into a 2D NumPy array
        shape = (30, 14)  # Desired shape
        array = np.array(cf_data).reshape(shape)

        return array

    def load_data(self, file_path):
        # Implement the logic to load and process the data from a text file
        # Return the processed data as a tensor or in the required format

        # with open(file_path, 'r') as file:
        #     data = file.read()h
        if file_path.endswith('.txt'):
            data = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
        elif file_path.endswith('.csv'): #When adding countefactual data into the training mix
            data = self.open_csv(file_path)

        #Convert data to tensor
        data_tensor = torch.tensor(data)

        return data_tensor
    #%%
