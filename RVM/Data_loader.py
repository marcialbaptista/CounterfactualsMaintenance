import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.txt')]

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

    def load_data(self, file_path):
        # Implement the logic to load and process the data from a text file
        # Return the processed data

        # with open(file_path, 'r') as file:
        #     data = file.read()h
        data = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)


        return data
    #%%
