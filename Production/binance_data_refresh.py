#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:18:12 2021

@author: bikramgill
"""
# %% libraries
import os
import pandas as pd
import numpy as np
import json
import requests
import datetime as dt
from binance.client import Client
import datetime as dt
import configparser
import pandas_ta as pta


# %% binance api info

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__),'config.txt'))
testnet_api= "https://testnet.binance.vision/api"

api_key = config['binance_testnet']['binance_testnet_api']
api_secret = config['binance_testnet']['binance_testnet_secret']

client = Client(api_key, api_secret)
client.API_URL=testnet_api


# %% read current data

class UpdateHistoricalData(object):
    def __init__(self, symbol, interval, start_date): 
        self.symbol = symbol
        self.interval = interval
        self.timestamp = start_date
        self.path = os.path.dirname(__file__) 
        
    
    def download_df(self):
        bars = client.get_historical_klines(self.symbol, self.interval, self.timestamp, limit=1000)
        new_df = pd.DataFrame(bars, columns = ['open_time','open','high','low','close','volume','close_time','quote_asset_vol','trade_count','taker_buy_vol', 'taker_sell_vol', 'ignore'])
        new_df['open_time'] = [dt.datetime.fromtimestamp(x/1000.0) for x in new_df['open_time']]
        new_df[['open','high','low','close','volume','quote_asset_vol','trade_count','taker_buy_vol', 'taker_sell_vol', 'ignore']] = new_df[['open','high','low','close','volume','quote_asset_vol','trade_count','taker_buy_vol', 'taker_sell_vol', 'ignore']].apply(pd.to_numeric)
        new_df['close_time'] = [dt.datetime.fromtimestamp(x/1000.0) for x in new_df['close_time']]
        new_df.set_index('close_time', inplace=True)
        new_df = new_df[['open','high','low','close','volume','trade_count']]
        self.new_df = new_df
        print("Processed new_df: " + self.symbol + "_" +self.interval)
        print("Download_DF, DF Columns", new_df.columns)
        return new_df
    
    def add_historical_df(self):
        binance_df = self.new_df
        historical_df = pd.read_csv(os.path.join(self.path, 'Datasets/', self.symbol + '_' + self.interval + '_Historical.csv'), parse_dates=['close_time'])
        historical_df.set_index('close_time', inplace=True)
        
        final_df = pd.concat([historical_df, binance_df])
        final_df = final_df.reset_index()
        final_df = final_df.drop_duplicates(subset=['close_time'], keep='first')
        final_df = final_df.set_index('close_time')
        self.new_df = final_df
        
        print("Added historical data")
    
    def add_features(self):
        ## Price vs Simple Moving Averages
        self.new_df['SMA50'] = self.new_df['close'] / self.new_df['close'].rolling(50).mean()
        self.new_df['SMA100'] = self.new_df['close'] / self.new_df['close'].rolling(100).mean()
        self.new_df['SMA200'] = self.new_df['close'] / self.new_df['close'].rolling(200).mean()
        
        ## RSI
        self.new_df['RSI'] = pta.rsi(self.new_df['close'], length=14)
        
        ## Close-to-Other Ratios
        self.new_df['Close_to_Open'] = self.new_df['close'] / self.new_df['open']
        self.new_df['Close_to_High'] = self.new_df['close'] / self.new_df['open']
        self.new_df['Close_to_Low'] = self.new_df['close'] / self.new_df['open']
        
        ## Volume and Trade Count Momentum
        self.new_df['Volume_Momentum'] = self.new_df['volume'] / self.new_df['volume'].rolling(10).mean()
        self.new_df['Trade_Count_Momentum'] = self.new_df['trade_count'] / self.new_df['trade_count'].rolling(10).mean()
        
        ## Create target variable based on next two candle analysis
        self.new_df['next_two_candles'] = 'Neutral'
        self.new_df.loc[(self.new_df['close'].shift(-1) > self.new_df['close']) & (self.new_df['close'].shift(-2) > self.new_df['close'].shift(-1)), 'next_two_candles'] = 'Up'
        self.new_df.loc[(self.new_df['close'].shift(-1) < self.new_df['close']) & (self.new_df['close'].shift(-2) < self.new_df['close'].shift(-1)), 'next_two_candles'] = 'Down'

        print("Added features to dataset: " + self.symbol + "_" +self.interval)
    


        
    def output_df(self):
        output_df = self.new_df
        output_df = output_df.reset_index() \
                             .drop_duplicates(subset=['close_time'], keep='first') \
                             .dropna() \
                             .set_index('close_time')
                             
        path = './Datasets/' + self.symbol + '_' + self.interval + '.csv'
        print("Current Path is {}".format(path))
        output_df.to_csv(path)
        print("Processed output_df: " + self.symbol + "_" +self.interval)


        
