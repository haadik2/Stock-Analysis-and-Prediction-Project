import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.index = pd.to_datetime(stock_data.index)  # Ensure index is DatetimeIndex
    return stock_data

def fetch_gdp_data():
    gdp_data = pd.read_csv('https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv')
    us_gdp = gdp_data[gdp_data['Country Name'] == 'United States'][['Year', 'Value']]
    us_gdp.rename(columns={'Value': 'GDP'}, inplace=True)
    us_gdp['Year'] = pd.to_datetime(us_gdp['Year'], format='%Y')
    return us_gdp

def merge_data(stock_data, gdp_data):
    stock_data['Year'] = stock_data.index.year
    gdp_data['Year'] = gdp_data['Year'].dt.year
    merged_data = stock_data.merge(gdp_data, left_on='Year', right_on='Year', how='left')
    merged_data.fillna(method='ffill', inplace=True)
    return merged_data
