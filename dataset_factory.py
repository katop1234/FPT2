import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import utils

'''
This script takes in the pandas df and then creates a custom dataset class that can be used with the DataLoader for 
parallelized training.

Steps:
Define the Categories and Windows:

Define your base categories like Open, High, Low, Close, Adj Close, and Volume.
Determine the window sizes as per your requirement (e.g., Open_Last_0_1_Days, Volume_Last_2516_3452_Days).
Create new columns in the DataFrame for each of these categories using the window sizes.
Preprocess the Data:

Iterate through the DataFrame and fill the newly created columns with the appropriate values from the existing columns based on the window sizes.
Apply padding with zeros to make all sequences have the same length (default_data_dim).
Ensure that the data is processed for all weekdays and has been checked for null/continuity.
Create a Custom PyTorch Dataset:

Define a custom dataset class that inherits from torch.utils.data.Dataset.
Initialize the dataset with the preprocessed DataFrame and the list of newly created columns (features).
Implement the __len__ and __getitem__ methods to return the size of the dataset and retrieve a specific item, respectively.
Create a PyTorch DataLoader:

Use the custom dataset to create a DataLoader, which can handle batching, shuffling, and parallel loading.
Set the batch size and other DataLoader parameters as needed.
Use the DataLoader in Training/Evaluation:

Iterate through the DataLoader in your training or evaluation loop, and it will yield batches of data that you can pass through your model.
Remember we don't want to randomly sample, but instead pass thru it over time.
'''

# Assuming utils.read is a function that reads the serialized data
df = utils.read("SNPdata.ser")
df = df.sort_values(by='Date')

import numpy as np
from pandas.tseries.offsets import BDay

floats_categories = utils.get_floats_categories()

# Define the data dimension (length of each token)
data_dim = 1280

# Get the float categories and time-related categories
floats_categories = utils.get_floats_categories()
time2vec_categories = utils.get_time2vec_categories()

# Preprocessing step to combine the windows and create attention masks
preprocessed_data = []
attention_masks = []
for i, row in df.iterrows():
    row_data = []
    row_mask = []
    date = row['Date']
    ticker = row['Ticker']

    # Handle the float categories
    for category in floats_categories:
        feature_name, start_days, end_days = category.split('_')
        start_days, end_days = int(start_days), int(end_days)
        
        start_date = date - BDay(start_days)
        end_date = date - BDay(end_days)
        
        window_data = df[(df['Date'] >= start_date) & (df['Date'] < end_date) & (df['Ticker'] == ticker)]
        window_values = window_data[feature_name].values
        
        if len(window_values) == 0:
            token_data = [-1] * data_dim  # Padding value
            mask_value = 0
        else:
            token_data = list(window_values) + [0] * (data_dim - len(window_values)) # Padding with zeros
            mask_value = 1
            
        row_data.append(token_data)
        row_mask.append(mask_value)
    
    # Handle the time-related categories
    for category in time2vec_categories:
        value = [float(row[category])]  # Convert to 1-length float
        token_data = value + [0] * (data_dim - 1) # Padding with zeros
        mask_value = 1 # Always include time-related categories
        
        row_data.append(token_data)
        row_mask.append(mask_value)

    preprocessed_data.append(row_data)
    attention_masks.append(row_mask)

preprocessed_data = torch.tensor(preprocessed_data, dtype=torch.float)  # Shape: [num_rows, N, D]
attention_masks = torch.tensor(attention_masks, dtype=torch.bool)      # Shape: [num_rows, N]

class CustomDataset(Dataset):
    def __init__(self, preprocessed_data, attention_masks):
        self.data = preprocessed_data
        self.masks = attention_masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.masks[idx]

batch_size = 64

# Create DataLoader
dataset = CustomDataset(preprocessed_data, attention_masks)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

for batch_idx, (data_batch, mask_batch) in enumerate(dataloader):
    data_batch, mask_batch = data_batch.cuda(), mask_batch.cuda()
    # data_batch is now of shape [B, N, D]
    # mask_batch is now of shape [B, N]
