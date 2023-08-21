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
target_values = []
# Set the multi-level index and sort
df = df.set_index(['Ticker', 'Date']).sort_index()

for (ticker, date), row in df.iterrows():
    row_data = []
    row_mask = []
    # Handle the float categories
    for category in floats_categories:
        feature_name, start_days, end_days, _ = category.split('_')
        start_days, end_days = int(start_days), int(end_days)
        
        start_date = date - BDay(start_days)
        end_date = date - BDay(end_days)
        
        # Use a loop to filter the DataFrame for the desired date range
        window_values = []
        for d in pd.date_range(start=start_date, end=end_date, freq='B'):
            try:
                value = df.loc[(ticker, d), feature_name]
                window_values.append(value)
            except KeyError:
                continue

        if not window_values:
            token_data = [-100] * data_dim  # Padding value
            mask_value = 0
        else:
            token_data = window_values + [0] * (data_dim - len(window_values)) # Padding with zeros
            mask_value = 1
            
        row_data.append(token_data)
        row_mask.append(mask_value)
    
    # Handle the time2vec categories
    ...
    
    # Append the ticker index also
    row_data.append([ticker] * data_dim)
    
    print("Row data shape: ", np.array(row_data).shape, "Attention mask shape: ", np.array(row_mask).shape, "for date and ticker: ", date, ticker)
    
    target_values.append(row['gt'])
    preprocessed_data.append(row_data)
    attention_masks.append(row_mask)

preprocessed_data = torch.tensor(preprocessed_data, dtype=torch.float).cuda()  # Shape: [num_rows, N, D]
attention_masks = torch.tensor(attention_masks, dtype=torch.bool).cuda()    # Shape: [num_rows, N]
targets = torch.tensor(target_values, dtype=torch.float).cuda()              # Shape: [num_rows, 1]

utils.write("preprocessed_data.ser", preprocessed_data)
utils.write("attention_masks.ser", attention_masks)
utils.write("targets.ser", targets)

class CustomDataset(Dataset):
    def __init__(self, preprocessed_data, attention_masks, targets):
        self.data = preprocessed_data
        self.masks = attention_masks
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.masks[idx], self.targets[idx]

batch_size = 64

# Create DataLoader
dataset = CustomDataset(preprocessed_data, attention_masks, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

