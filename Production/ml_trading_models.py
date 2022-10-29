#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 08:11:38 2022

@author: bikramgill
"""
# %% Import libraries

import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier as xgb
from sklearn.linear_model import LogisticRegression as lgr
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold

# %% Model Class

class model_builder(object):
    def __init__(self, model_type, symbol, interval): 
        self.model = model_type
        self.symbol = symbol
        self.interval = interval
        path = os.path.dirname(__file__) 
        self.filepath = os.path.join(path, 'Datasets/', symbol + '_' + interval + '.csv')
        self.df = pd.read_csv(self.filepath)
        
    
    def train_model_pickle(self):
        self.X_train = self.new_df.iloc[:-50, :].drop(columns='close').copy()
        self.X_test =  self.new_df.iloc[-50:, :].drop(columns='close').copy()
        
        self.y_train = self.new_df.iloc[:-50, :]['close'].copy()
        self.y_test = self.new_df.iloc[-50:, :]['close'].copy()
        
        print("Created Train/Test Split.\nX_train: {} records. \nX_test: {} records.".format(len(self.X_train), len(self.X_test)))
        

    
    def add_features(self):
        self.new_df['SMA50'] = self.new_df['close'].rolling(50).mean()
        self.new_df['SMA100'] = self.new_df['close'].rolling(100).mean()
        self.new_df['SMA200'] = self.new_df['close'].rolling(200).mean()

