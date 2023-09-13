import numpy as np
import pandas as pd
import torch
import utils
from torch.utils.data import Dataset

date_ranges = utils.get_date_ranges()
base_categories = utils.base_categories()
time2vec_categories = utils.get_time2vec_categories()
token_length = 2 ** utils.POWER_OF_2 # Dimensionality of token
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
            return (None, None, None)
        X = np.empty((0, token_length))
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
                    # TODO is it ok to pad with 0s since those are within the range of values for returns?
                    padding = np.zeros((array.shape[0], token_length-array.shape[1]))
                    array = np.hstack((array, padding))

            X = np.vstack((X, array))
        
        y = self.gt[self.counter]
        
        # TODO you may have to normalize the input values for consistency, altho layernorm should fix that

        self.counter += 1
        return X, mask, y

class FinancialDataset(Dataset):
    def __init__(self, df_path):
        self.df = utils.read(df_path)
        self.ticker_names = self.df["Ticker"].unique().tolist()
        self.df = self.df.set_index(["Ticker"])
        # each ticker needs start date, num_rows, numpy array of input features, gt vector
        self.ticker_metadata = {}
        self.ticker_generator_lookup = {}

        for ticker in self.ticker_names:
            metadata = self.add_ticker_metadata(ticker)
            self.ticker_generator_lookup[ticker] = Ticker(metadata)

        self.stack = self.ticker_names
    
    def pop_stack_and_move_to_back(self):
        popped = self.stack[0]
        self.stack = self.stack[1:].append(self.stack[0])
        return popped

    def add_ticker_metadata(self, ticker):
        ticker_df = self.df.loc[ticker]
        ticker_df = ticker_df.sort_values(by="Date")
        start_date = ticker_df["Date"].iloc[0]
        end_date = ticker_df["Date"].iloc[-1]
        num_rows = len(ticker_df)
        ticker_df = ticker_df.reset_index()
        ticker_df = ticker_df.drop(columns=["Date", "TypicalPrice", "Ticker"])
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
        return metadata

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # call gen function for next ticker in stack
        
        # TODO add code for when all the tickers are done (i.e. stack is empty)
        
        sample = None
        while sample is None:
            chosen_ticker = self.pop_stack_and_move_to_back()
            Ticker_generator = self.ticker_generator_lookup[chosen_ticker]
            sample, mask, y = Ticker_generator.gen_function()
            if sample is None:
                # done with that ticker
                self.stack = self.stack[:-1]
            else:
                x = torch.from_numpy(sample).float()
                
                mask = [True] + mask # for CLS
                mask = torch.tensor(mask, dtype=torch.bool)
                
                y = torch.tensor(y, requires_grad=True).unsqueeze(0).unsqueeze(0)
                
                return x, mask, y # TODO convert to torch during init and figure out how to feed in mask
