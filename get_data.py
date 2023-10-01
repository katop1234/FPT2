import os.path

import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
from utils import read, write
from torch.utils.data import Dataset
import torch

def get_SNP_list():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table_id = 'constituents'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    constituents_table = soup.find('table', attrs={'id': table_id})
    df = pd.read_html(str(constituents_table))[0]
    return df['Symbol'].tolist()

def get_single_ticker_data_yfinance(ticker):
    '''
    Returns a dataframe of a single ticker's available data
    Columns are: Open, High, Low, Close, Adj Close, Volume, Ticker, Year, Month, Day
    '''
    # set the start date to the earliest possible date
    data = yf.download(ticker, start='1970-01-01', end='2020-01-01')
    
    # Ensure that the dates are within the expected range
    if not data.empty and ((data.index < pd.Timestamp('1970-01-01')) | (data.index > pd.Timestamp('2022-01-01'))).any():
        raise ValueError(f"Data for {ticker} contains dates outside the expected range.")
    
    data['Ticker'] = ticker
    data.reset_index(inplace=True)
    
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.dayofweek % 7
    
    data['TypicalPrice'] = data[['High', 'Low', 'Close']].mean(axis=1)

    return data

def get_all_SNP500_data():
    '''
    Returns a dataframe of each SNP ticker's last 20 years of data
    '''
    # First, get the list of all S&P 500 tickers
    tickers = get_SNP_list()

    # Initialize an empty dataframe to hold all the data
    all_data = pd.DataFrame()

    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}")
            data = get_single_ticker_data_yfinance(ticker)
            all_data = pd.concat([all_data, data], axis=0)
        except Exception as e:
            print(f"Could not fetch data for {ticker}. Reason: {e}")
            continue

    return all_data

def calculate_gt(group):
    '''
    Use this to get the gt column
    Apply function to each ticker group
    '''
    group['shifted_price'] = group['TypicalPrice'].shift(1)
    group['gt'] = (group['TypicalPrice'] - group['shifted_price']) / group['shifted_price']
    
    # Set the first value of gt for each group to 0
    group['gt'].iloc[0] = 0

    group.drop(columns=['shifted_price'], inplace=True)
    return group

import pandas as pd
from tqdm import tqdm

def normalize_column(df):
    '''
    Use this to normalize specified columns within each ticker group.
    '''
    normalizing_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    
    def normalize(group):
        # Compute the relative change for specified columns
        group[normalizing_cols] = group[normalizing_cols].pct_change().fillna(0)
        return group

    # Wrap the groupby object with tqdm for progress bar
    tqdm.pandas()
    return df.groupby('Ticker').progress_apply(normalize)

def normalize_column(df):
    '''
    Use this to normalize specified columns within each ticker group.
    '''
    normalizing_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    
    def normalize(group):
        # Compute the relative change for specified columns
        group[normalizing_cols] = group[normalizing_cols].pct_change().fillna(0)
        return group

    # List to collect processed DataFrames for each ticker
    processed_data = []

    # Iterate over each ticker, apply the normalize function, and collect the result
    for ticker, group in tqdm(df.groupby('Ticker'), desc="Processing tickers"):
        processed_data.append(normalize(group))

    # Combine the processed data together
    return pd.concat(processed_data)

def write_all_SNP500_data():
    '''
    Writes all SNP500 data to a file
    '''
    df = get_all_SNP500_data()
    
    # Apply the calculate_gt function to each ticker group
    df = df.groupby('Ticker').apply(calculate_gt).reset_index(drop=True)
    
    # Convert 'Date' to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Resample to include all weekdays and interpolate missing values
    def resample_ticker_group(group):
        group = group.set_index('Date')
        group = group.resample('B').asfreq()
        group.interpolate(method='linear', inplace=True)
        return group.reset_index() # Reset the index to keep 'Date' as a column

    df = df.groupby('Ticker', group_keys=False).apply(resample_ticker_group)

    # Ensure that the 'Ticker' column is of string type
    df['Ticker'] = df['Ticker'].astype(str)

    # fill in any missing weekday values
    num_nans_before = df.isna().sum().sum()
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    print(f"Filled {num_nans_before} missing values with forward and backward fill.")

    df = df[df['Ticker'] != 0]
    
    # Normalize datetime columns
    df["Year"] = (df["Year"] - 1900) / 150 
    df["Month"] = (df["Month"] - 1) / 12 # [1, 12]
    df["Day"] = (df["Day"] - 1) / 31 # [1, 31]
    df["Weekday"] = df["Weekday"] / 5 # [0, 4]

    write(df, "SNPdata.ser")


'''
DATA PREPROCESSING:
- filling in any missing weekday values
df.set_index('Date', inplace=True)
# Resample to include all weekdays and interpolate missing values
df = df.resample('B').asfreq()  # 'B' stands for business day
df.interpolate(method='linear', inplace=True)
'''

def write_single_ticker_data_yfinance():
    all_data = read("SNPdata.ser")
    if not os.path.exists("./serialized/ticker_data/"):
        os.mkdir("./serialized/ticker_data/")
    for ticker in all_data["Ticker"].unique():
        try:
            ticker_data = get_single_ticker_data_yfinance(ticker)
            write(ticker_data, "ticker_data/" + ticker + "_data.ser")
        except Exception as e:
            print(f"Could not fetch data for {ticker}. Reason: {e}")
            continue

write_all_SNP500_data()
#write_single_ticker_data_yfinance()
