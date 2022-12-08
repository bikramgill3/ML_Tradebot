#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 17:54:39 2022

@author: bikramgill

"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder 
import os
from sklearn.model_selection import train_test_split
path = os.path.dirname(__file__) 


def split_data(model, symbol, interval, start_date, end_date):
   
    print("Loading Data for Train/Test Split")
    ## Load df and cut down to start and end period defined
    df = pd.read_csv(os.path.join(path, 'Datasets/', symbol + '_' + interval + '.csv'), parse_dates = ['close_time']).dropna()
    df = df.loc[(df['close_time']>=start_date) & (df['close_time']<=end_date)]
    
    print("Define feature and target columns for X and y")
    ## Define feature and target columns for X and y
    feature_cols = ['SMA50', 'SMA100', 'SMA200', 'RSI', 'Close_to_Open', 'Close_to_High', 'Close_to_Low', 'Volume_Momentum', 'Trade_Count_Momentum']
    target_col = ['next_two_candles']
    X = df[feature_cols]
    y = df[target_col]
    
    print("Convert text labels with LabelEncoder")
    ## Convert text labels with LabelEncoder
    lc = LabelEncoder()
    y = lc.fit_transform(y)
    
    print("Conduct test train split")
    ## Conduct test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    return(X_train, X_test, y_train, y_test)