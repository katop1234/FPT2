import numpy as np
import pandas as pd
import torch
import utils
from torch.utils.data import Dataset

date_ranges = utils.get_date_ranges()
base_categories = utils.base_categories()
time2vec_categories = utils.get_time2vec_categories()
token_length = 2 ** utils.POWER_OF_2
num_categories = len(base_categories) + len(time2vec_categories)
num_tokens = len(date_ranges) * num_categories
class Ticker:
    def __init__(self, metadata):
        self.ticker = metadata["ticker"]
        # self.start_date = metadata["start_date"]
        # self.end_date = metadata["end_date"]
        self.num_rows = metadata["num_rows"]
        self.raw_data = metadata["raw_data"]
        self.gt = metadata["gt"]

        self.counter = 365 # TODO make hyperparameter

    # one datapoint, 200 tokens of length 256
    # 20 time windows, of 10 categories
    def gen_function(self):
        if self.counter >= self.num_rows:
            return None
        datapoint = np.empty((0, token_length))
        mask = []
        for start, end in date_ranges:
            if self.counter - end < 0:
                mask += [False] * num_categories
                array = np.empty((num_categories, token_length))
            else:
                mask += [True] * num_categories
                array = self.raw_data[self.counter-end:self.counter-start]
                array = array.T
                if array.shape[1] < token_length:
                    np.hstack(array, np.zeros((array.shape[0], token_length-array.shape[1])))

            datapoint = np.vstack(datapoint, array)

        self.counter += 1
        return datapoint, mask

class FinancialDataset(Dataset):
    def __init__(self, df_path):
        self.df = utils.read(df_path)
        self.ticker_names = self.df["Ticker"].unique()
        self.df.set_index(["Ticker", "Date"])
        # each ticker needs start date, num_rows, numpy array of input features, gt vector
        self.ticker_metadata = {}

        for ticker in self.ticker_names:
            self.add_ticker_metadata(ticker)



        # by the end of the init method return a tensor of shape
        # (num_features=num_categories*num_windows=10*20, max_dim=256)

    def add_ticker_metadata(self, ticker):
        ticker_df = self.df.loc[ticker]
        start_date = ticker_df["Date"][0]
        end_date = ticker_df["Date"][-1]
        num_rows = len(ticker_df)
        ticker_df = ticker_df.reset_index()
        ticker_df = ticker_df.drop(columns=["Date", "TypicalPrice"])
        np_array = ticker_df.to_numpy()
        gt_vector, raw_data = np_array[:, -1], np_array[:, :-1]

        metadata = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "num_rows": num_rows,
            "raw_data": raw_data,
            "gt": gt_vector,
                    }

        self.ticker_metadata[ticker] = metadata

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # call gen function for next ticker in stack

        return sample

# Example usage:
dataset = FinancialDataset("path_to_your_data.csv")
sample = dataset[0]  # Fetch the first sample from the dataset