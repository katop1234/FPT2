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

# Your code for base_categories() and get_floats_categories() here...

# Assuming df is your DataFrame
categories = utils.get_floats_categories()
default_data_dim = 1280

# Iterate through categories to create and populate new columns
for category in categories:
    base_cat, start_window, end_window = category.split('_')[0], int(category.split('_')[1]), int(category.split('_')[-2])
    col_name = f"{base_cat}_{start_window}_{end_window}_days"
    df[col_name] = 0  # Initialize column with zeros

    # Populate column values based on window and base category
    for index, row in df.iterrows():
        if index >= end_window:
            values = df[base_cat][index - end_window:index - start_window].values
            if len(values) < default_data_dim:
                values = [0] * (default_data_dim - len(values)) + list(values)
            df.at[index, col_name] = values  # Assuming that the values are stored as a list or similar structure


# Create a custom dataset class
class FinancialDataset(Dataset):
    def __init__(self, dataframe, input_features, target_feature):
        self.dataframe = dataframe
        self.input_features = input_features
        self.target_feature = target_feature

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        inputs = torch.tensor([self.dataframe[feature][idx] for feature in self.input_features], dtype=torch.float32)
        target = torch.tensor(self.dataframe[self.target_feature][idx], dtype=torch.float32)
        return inputs, target

# List of input feature columns, you can modify as needed
input_features = categories  # Categories we created earlier
target_feature = 'gt'  # Modify this as needed

# Create the custom dataset
dataset = FinancialDataset(df, input_features, target_feature)

# Create the DataLoader
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Example of loading data to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for batch_inputs, batch_targets in dataloader:
    batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
    # Your model training or evaluation code here
