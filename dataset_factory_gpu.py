import utils
import torch
import pandas as pd
import sys

'''
ok the efficient idea would be to have all the ticker-specific tensors ready in memory then 
iterate over each row of the df, and for each one, append the desired window tensor
'''

# Hyperparameters
data_dim = 1280
# Hyperparameters

df = utils.read("SNPdata.ser")
df = df[df['Ticker'] != '0']
df = df[df['Date'] != '0']
df = df[df['Ticker'].str.isalnum()]

base_categories = utils.get_base_categories()
floats_categories = utils.get_floats_categories()
time2vec_categories = utils.get_time2vec_categories()

tensor_categories = base_categories + time2vec_categories

# Grouping the DataFrame by the 'Ticker' column
grouped = df.groupby('Ticker')

# Dictionary to store the tensor data for each ticker symbol
all_tensor_data = {}

# Iterate through the groups (each group corresponds to a specific ticker)
for ticker, group in grouped:
    # Sort the group by the 'Date' column
    sorted_group = group.sort_values(by='Date')
    
    # Select the columns corresponding to the base categories
    ticker_data = sorted_group[tensor_categories]
    
    # Convert the DataFrame for the current ticker into a PyTorch tensor
    ticker_tensor = torch.tensor(ticker_data.values, dtype=torch.float32)
    
    # Add the tensor to the dictionary with the ticker symbol as the key
    all_tensor_data[ticker] = ticker_tensor
    
    print(ticker)

# You can now access the tensor for a specific ticker using all_tensor_data[ticker_symbol]
# For example:
ticker_to_use = "AAPL"
print("Shape of", ticker_to_use, "tensor is", all_tensor_data[ticker_to_use].shape)  # Output should be [num_rows, len(base_categories)]

final_tensor = []

# Print the shape of the final tensor
print(final_tensor.shape)  # Output shape: [total_num_rows_across_all_tickers, len(floats_categories) + len(time2vec_categories), D]

utils.write(final_tensor, "final_tensor.pt")
