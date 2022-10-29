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


# %% binance api info

config = configparser.ConfigParser()
config.read('/Users/bikramgill/Documents/Documents/config.txt')
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
        path = os.path.dirname(__file__) 
        self.filepath = os.path.join(path, 'Datasets/', symbol + '_' + interval + '.csv')
        
    
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
        return new_df
    
    def add_features(self):
        ## Set of Simplee Moving Averages
        self.new_df['SMA50'] = self.new_df['close'] / self.new_df['close'].rolling(50).mean()
        self.new_df['SMA100'] = self.new_df['close'] / self.new_df['close'].rolling(100).mean()
        self.new_df['SMA200'] = self.new_df['close'] / self.new_df['close'].rolling(200).mean()
        
        ## Create target variable based on next two candle analysis
        self.new_df['next_two_candles'] = 'Neutral'
        self.new_df.loc[(self.new_df['close'].shift(-1) > self.new_df['close']) & (self.new_df['close'].shift(-2) > self.new_df['close'].shift(-1)), 'next_two_candles'] = 'Up'
        self.new_df.loc[(self.new_df['close'].shift(-1) < self.new_df['close']) & (self.new_df['close'].shift(-2) < self.new_df['close'].shift(-1)), 'next_two_candles'] = 'Down'

        print("Added features to dataset: " + self.symbol + "_" +self.interval)
    


        
    def output_df(self):
        output_df = self.new_df
        output_df = output_df.reset_index()
        output_df = output_df.drop_duplicates(subset=['close_time'], keep='first')
        output_df = output_df.set_index('close_time')
        output_df.to_csv(self.filepath, header=True)
        print("Processed output_df: " + self.symbol + "_" +self.interval)


        
