import os.path

import requests
from bs4 import BeautifulSoup
import yfinance as yf
from utils import read, write
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm


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
    start_date, end_date = '1970-01-01', '2020-01-01'

    # set the start date to the earliest possible date
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Ensure that the dates are within the expected range
    if not data.empty and ((data.index < pd.Timestamp(start_date)) | (data.index > pd.Timestamp(end_date))).any():
        raise ValueError(f"Data for {ticker} contains dates outside the expected range.")
    
    data['Ticker'] = ticker
    data.reset_index(inplace=True)
    
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

def percent_diff(group, col_name):
    '''
    Use this to get the gt column
    Apply function to each ticker group
    '''
    group['shifted_col'] = group[col_name].shift(1)
    group['shifted_col'].iloc[0] = 1 # dummy value, we drop the first row anyway

    num_nans = group['shifted_col'].isna().sum()
    if num_nans > 0:
        print("WARNING got {num_nans} nans in column", col_name, "for ticker", group.index)

    group[col_name] = (group[col_name] - group['shifted_col']) / group['shifted_col']

    group[col_name].fillna(0, inplace=True)
    
    group.drop(columns=['shifted_col'], inplace=True)

    group.drop(group.index[0], inplace=True) # drop first row

    return group

def write_all_SNP500_data():
    '''
    Writes all SNP500 data to a file
    '''
    df = get_all_SNP500_data()

    print("Got all SNP 500 data!")
    list_of_cols_to_normalize = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "TypicalPrice"]

    # Apply the calculate_gt function to each ticker group
    for col in list_of_cols_to_normalize:
        df = df.groupby('Ticker').apply(lambda group: percent_diff(group, col)).reset_index(drop=True)
    
    df.rename(columns={'TypicalPrice': 'gt'}, inplace=True)
    # Convert 'Date' to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])
    print("Converted Date column to pd datetime")
    
    # Resample to include all weekdays and interpolate missing values
    def resample_ticker_group(group):
        group = group.set_index('Date')
        ticker_value = group['Ticker'].iloc[0] 
        group = group.resample('B').asfreq()
        group.interpolate(method='linear', axis=0, inplace=True)
        group['Ticker'].fillna(ticker_value, inplace=True)
        return group.reset_index() # Reset the index to keep 'Date' as a column

    df = df.groupby('Ticker', group_keys=False).apply(resample_ticker_group)
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.dayofweek % 7

    df["Year"] = (df["Year"] - 1900) / 150 
    df["Month"] = (df["Month"] - 1) / 12 # [1, 12]
    df["Day"] = (df["Day"] - 1) / 31 # [1, 31]
    df["Weekday"] = df["Weekday"] / 5 # [0, 4]

    write(df, "SNPdata.ser")
    print("Wrote df!")

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

# write_all_SNP500_data()
