#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 13:47:44 2022

@author: bikramgill
"""

# %% import libraries

import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import base64
import os
from datetime import date, datetime as dt
from dateutil.relativedelta import relativedelta

from binance_data_refresh import UpdateHistoricalData as UHD
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
from sklearn.neighbors import KNeighborsClassifier
import time
import model_retrain
from plotly.graph_objs import *



# %% Notes and To-Dos

# 1) Change color palette and use the color codes you found on instagram


# %% set OS filepath

path = os.path.dirname(__file__) 
print(path)

# %% dash app setup

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=False
)

app.title = "USD Capstone - Tradebot" # puts title on browser tab

# %% styling dictionaries

main_style = {
    'background': '#C3C3D5',
    'font-family': "Verdana",
    'height': '100vh'
}

title_style = {
    'color': 'black',
    'valign': 'top'  
}

dropdown_style = {
    'background': 'grey',
    'text': 'grey' 
}

text_center = {'textAlign': 'center'
}

text_left = {'textAlign': 'left'
}

vert_mid = {'vertical-align': 'middle'
}

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

date_picker_style = {

}


button_style = {
    'align': 'center'
}


# %% Functions for tab content based on selected tab. Used in app callback to render content. 
def market_analysis_tab():
    return html.Div([
            
        html.Div([
            ## Analysis Div 1.1: Date Selector and header.
            html.Div([
                html.Label('Start/End Date Selector', style = title_style),
                dcc.DatePickerRange(
                    id='market_analysis_date_picker',
                    min_date_allowed = date.today() + relativedelta(years=-4),
                    max_date_allowed = date.today() + relativedelta(days=-1),
                    initial_visible_month = date.today() + relativedelta(days=-1),
                    start_date = date.today() + relativedelta(years=-4),
                    end_date=date.today() + relativedelta(days=-1)
                    )
                ], style={'display': 'inline-block', 'width':'25%'}),
    
            
            ## Analysis Div 1.2: Market Selector and header.
            html.Div([
                html.Label('Market Selector', style = title_style),
                dcc.Dropdown(['BTCUSDT', 'ETHUSDT'], 'BTCUSDT', id='market_selector', clearable=False)
                ], style={'display': 'inline-block', 'width':'25%'}),
            
            ## Analysis Div 1.3: Market Selector and header.
            html.Div([
                html.Label('Tick Interval (Hours)', style = title_style),
                dcc.Dropdown(['1h', '4h'], '4h', id='market_interval', clearable=False)
                ], style={'display': 'inline-block', 'width':'25%'}),
            
            
            ## Analysis Div 1.4: Update button
            html.Div([
                html.Button('Update Graph', id='update_market_analysis', n_clicks=0)
                ], style={'display': 'inline-block', 'width':'10%'})
        ]),
        
        
        html.Br(),

        html.Div(id='market_graph'),
        
        html.Br(),

        html.Div([html.Button('Download Data', id='download_market_data', n_clicks=0)
                  ])
        
    ])



def ml_model_tab():
    return html.Div([

        html.Div([
            ## Model Div 1.1: Model Information label. 
            html.Div([
                html.H3('Model Information', style = title_style | text_center),
                
                ## Model Div 1.1.1: Model Type Selector and label
                html.Div([
                    html.Label('Model Type', style = title_style | text_center),
                    dcc.Dropdown(['XGBoost', 'KNN'], 'KNN', id='model_type', clearable=False)
                    ], style={'display': 'inline-block', 'width':'33%'}),
                
                ## Model Div 1.1.2: Model Market Selector and label
                html.Div([
                    html.Label('Market', style = title_style | text_center),
                    dcc.Dropdown(['BTCUSDT', 'ETHUSDT'], 'BTCUSDT', id='model_market', clearable=False)
                    ], style={'display': 'inline-block', 'width':'33%'}),
                
                ## Model Div 1.1.3: Model Interval Selector and label
                html.Div([
                    html.Label('Tick Interval (Hours)', style = title_style | text_center),
                    dcc.Dropdown(['1h', '4h'], '4h', id='model_interval', clearable=False)
                    ], style={'display': 'inline-block', 'width':'33%'}),
            
                html.Div([html.Br()]),
                
                ## Div 1.1.4 Model Re-Train button
                html.Div([html.Button('Re-Train Model', id='model_train', n_clicks=0)], style=button_style | text_center),
                
                html.Div([html.Br()]),
                
                
                html.H3('Model Performance', style = title_style | text_center),
            
                html.Div(id='model_results_heatmap', children=[], style={'display': 'inline-block', 'width':'50%'}),
                
                html.Div(id='model_results_table', children=[], style={'display': 'inline-block', 'width':'50%'}),
                
                
                
                
                ], style={'display': 'inline-block', 'width':'100%'})
            
            ])
        ])


def trading_tab():
    return html.Div([

        html.Div([
            ## Model Div 1.1: Model Information label. 
            html.Div([
                html.H3('Model Settings', style = title_style | text_center),
                
                ## Model Div 1.1.1: Model Type Selector and label
                html.Div([
                    html.Label('Model Type', style = title_style | text_center),
                    dcc.Dropdown(['XGBoost', 'KNN'], 'KNN', id='tr_model_type', clearable=False)
                    ], style={'display': 'inline-block', 'width':'33%'}),
                
                ## Model Div 1.1.2: Model Market Selector and label
                html.Div([
                    html.Label('Market', style = title_style | text_center),
                    dcc.Dropdown(['BTCUSDT', 'ETHUSDT'], 'BTCUSDT', id='tr_model_market', clearable=False)
                    ], style={'display': 'inline-block', 'width':'33%'}),
                
                ## Model Div 1.1.3: Model Interval Selector and label
                html.Div([
                    html.Label('Tick Interval (Hours)', style = title_style | text_center),
                    dcc.Dropdown(['1h', '4h'], '4h', id='tr_model_interval', clearable=False)
                    ], style={'display': 'inline-block', 'width':'33%'}),
            
                html.Div([html.Br()]),
                
                ## Div 1.1.4 Model Re-Train button
                html.Div([html.Button('Predict Next Two Intervals', id='tr_prediction', n_clicks=0)], style=button_style | text_center),
                
                html.Div([html.Br()]),
                
                
                html.H3('Model Prediction', style = title_style | text_center),
            
                html.Div(id='tr_prediction_result', children=[])             
                
                ], style={'display': 'inline-block', 'width':'100%'})
            
            ])
        ])



# %% images
robot_logo = os.path.join(path, 'robot_logo.jpg')
robot_logo_enc = base64.b64encode(open(robot_logo, 'rb').read())



# %% define app layout

app.layout = html.Div([         # parant div
                       
    ## Div 1: Title Banner
    html.Div([
        
        ## Div 1.1: Robot Logo
        html.Div([
            html.Img(src='data:image/jpg;base64,{}'.format(robot_logo_enc.decode()),
                     width='100',
                     height='120')
            ], style={'display': 'inline-block'}),
        
        ## Div 1.2: USD Capstone Header
        html.Div([
            html.H3(children="USD Capstone - Machine Learning Tradebot",
                    style=title_style | text_center)
            ], style={'display': 'inline-block'})
        
        ]),
    
    ## Div 2: Tabs
    html.Div([
        dcc.Tabs(id='main_tabs', value='tab_analysis',
                 children=[
                     dcc.Tab(label='Trading', value='tab_trading', style=tab_style, selected_style=tab_selected_style, children=trading_tab()),
                     dcc.Tab(label='Market Analysis', value='tab_analysis', style=tab_style, selected_style=tab_selected_style, children=market_analysis_tab()),
                     dcc.Tab(label='ML Model Hub', value='tab_model', style=tab_style, selected_style=tab_selected_style, children=ml_model_tab()),
                     ])
        
        
        ]),
    
    ## Div 3: Build tab content.
    
    ], style=main_style)                           # close parent div



# %% Market Analysis Callback - Update Graph and Datasets
@app.callback(
    Output('market_graph', 'children'),
    Input('update_market_analysis', 'n_clicks'),
    State('market_analysis_date_picker', 'start_date'),
    State('market_analysis_date_picker', 'end_date'),
    State('market_selector', 'value'),
    State('market_interval', 'value')
)
def market_analytics_update_page(n_clicks, start_date, end_date, symbol, interval):
    
    last_update = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    interval=interval
    binance_data = UHD(symbol, interval, start_date)
    binance_data.download_df()
    binance_data.add_historical_df()
    binance_data.add_features()
    binance_data.output_df()

    
    graph_df = pd.read_csv(os.path.join(path, 'Datasets/', symbol + '_' + interval + '.csv'))
    graph_df['close_time'] = pd.to_datetime(graph_df['close_time'])

    graph_df = graph_df[graph_df['close_time'] >= start_date]
        

    
    fig = px.line(graph_df, 
                  x="close_time", 
                  y="close", 
                  title = 'Time Period: {}-{}. Market: {}. Last Updated: {}.'.format(start_date, end_date, symbol, last_update)
                  )
    
    return dcc.Graph(
    figure={
        'data': [
            {'x': graph_df['close_time'], 'y': graph_df['close'], 'type': 'line', 'name': 'SF'}
        ],
        'layout': {
            'title': 'Time Period: {}-{}. Market: {}. Last Updated: {}.'.format(start_date, end_date, symbol, last_update)
        }
    }
)


# %% ML Model Hub Callback - Re-train model
@app.callback(
    Output('model_results_heatmap', 'children'),
    Output('model_results_table', 'children'),
    Input('model_train', 'n_clicks'),
    State('model_type','value'),
    State('model_market','value'),
    State('model_interval','value'),
    State('market_analysis_date_picker', 'start_date'),
    State('market_analysis_date_picker', 'end_date'),
    prevent_initial_call=True
)

def update_model(n_clicks, model, symbol, interval, start_date, end_date):    
    print("Updating Model: {}_{}_{}".format(model,symbol, interval))
    
    X_train, X_test, y_train, y_test = model_retrain.split_data(model, symbol, interval, start_date, end_date)
    
    
    if model == 'KNN':
        results = []
        
        
        print("KNN Model Step 1")
        for n_neighbors in range(1, 20):
            
            print("Fit KNN")
            knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
            
            print("KNN Preds")
            preds = knn.predict(X_test)
            
            print("KNN Results")
            results.append({'n_neighbors': n_neighbors,
                            'accuracy': accuracy_score(y_test, preds),
                            'F1':f1_score(y_test, preds, average='macro')
                            })
            
        print("KNN Model Step 2")
        # Convert results to a pandas data frame results = pd.DataFrame(results) print(results)
        results = pd.DataFrame(results)
        results = results.sort_values(by='F1', ascending=False)
        
        n_neighbors = int(results.loc[results['F1'].idxmax()]['n_neighbors'])
        
        knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)

        print("KNN Model Step 3")
        with open(os.path.join(path,'Models/{}/KNN_{}_{}.pickle'.format(model, symbol, interval)), 'wb') as model:
            pickle.dump(knn, model, protocol=pickle.HIGHEST_PROTOCOL)
            
        #### Create Confusion Matrix
            
        x = ['Down', 'Neutral', 'Up']
        y = ['Down', 'Neutral', 'Up']
        fig = px.imshow(confusion_matrix(y_test, knn.predict(X_test)), 
                        labels=dict(x="Predicted Values", y="Actual Values"),
                        x=x, 
                        y=y,
                        text_auto=True)
        
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        
        #### Create Metrics Table
        
        report = classification_report(y_test, knn.predict(X_test), target_names=['Down', 'Neutral', 'Up'], output_dict=True)
        df1 = pd.DataFrame.from_dict(report)[['Down', 'Neutral','Up']]
        df1
        
        df2 = pd.DataFrame.from_dict(report)[['accuracy', 'macro avg','weighted avg']] \
                .reset_index(drop=True) \
                .transpose() \
                .drop(columns=3) \
                .rename(columns={0:'Down', 1:'Neutral', 2:'Up'})
        
        df2
        
        final_report = pd.concat([df1, df2])
        
        for i in final_report.columns:
            final_report[i] = final_report[i].round(3)
        
        final_report = final_report.reset_index().rename(columns={'index':'Metric'})
            
    
    elif model == "XGBoost":
        
        #### Run Model
        
        xgboost = XGBClassifier(random_state=1,
                            booster='gbtree',
                            learning_rate=0.1,
                            max_depth=5,
                            objective='multi:softmax')
        
        
        xg_boost = xgboost.fit(X_train,y_train)
        
        with open(os.path.join(path,'Models/{}/XGBoost_{}_{}.pickle'.format(model, symbol, interval)), 'wb') as model:
            pickle.dump(xg_boost, model, protocol=pickle.HIGHEST_PROTOCOL)
            
        #### Create Confusion Matrix
            
        x = ['Down', 'Neutral', 'Up']
        y = ['Down', 'Neutral', 'Up']
        fig = px.imshow(confusion_matrix(y_test, xg_boost.predict(X_test)), 
                        labels=dict(x="Predicted Values", y="Actual Values"),
                        x=x, 
                        y=y,
                        text_auto=True)
        
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        
        #### Create Metrics Table
        
        report = classification_report(y_test, xg_boost.predict(X_test), target_names=['Down', 'Neutral', 'Up'], output_dict=True)
        df1 = pd.DataFrame.from_dict(report)[['Down', 'Neutral','Up']]
        df1
        
        df2 = pd.DataFrame.from_dict(report)[['accuracy', 'macro avg','weighted avg']] \
                .reset_index(drop=True) \
                .transpose() \
                .drop(columns=3) \
                .rename(columns={0:'Down', 1:'Neutral', 2:'Up'})
        
        df2
        
        final_report = pd.concat([df1, df2])
        
        for i in final_report.columns:
            final_report[i] = final_report[i].round(3)
        
        final_report = final_report.reset_index().rename(columns={'index':'Metric'})
        
        
        
    print("Updated model and saved as pickle file.")
    
    return [html.H3('Model Results: Heatmap', style = title_style | text_center), dcc.Graph(figure=fig)], [html.H3('Model Results: Metrics Table', style = title_style | text_center), dash_table.DataTable(final_report.to_dict('records'), [{"name": i, "id": i} for i in final_report.columns])]


# %% Trading Callback - predict next two candles

@app.callback(
    Output('tr_prediction_result', 'children'),
    Input('tr_prediction', 'n_clicks'),
    State('tr_model_type','value'),
    State('tr_model_market','value'),
    State('tr_model_interval','value'),
    State('market_analysis_date_picker', 'start_date'),
    prevent_initial_call=True
)

def update_model(n_clicks, model, symbol, interval, start_date):
    interval=interval
    binance_data = UHD(symbol, interval, start_date)
    binance_data.download_df()
    binance_data.add_historical_df()
    binance_data.add_features()
    binance_data.output_df()
    
    latest_candle_df = pd.read_csv(os.path.join(path, 'Datasets/', symbol + '_' + interval + '.csv'))
    latest_candle_df['close_time'] = pd.to_datetime(latest_candle_df['close_time'])
    max_candle_time = latest_candle_df['close_time'].max()
    max_candle_data = latest_candle_df.loc[latest_candle_df['close_time'] == max_candle_time]
    feature_cols = ['SMA50', 'SMA100', 'SMA200', 'RSI', 'Close_to_Open', 'Close_to_High', 'Close_to_Low', 'Volume_Momentum', 'Trade_Count_Momentum']
    max_candle_cols = max_candle_data[feature_cols]
    
    if model == "KNN":
        with open((path + '/Models/{}/KNN_{}_{}.pickle'.format(model, symbol, interval)), 'rb') as models:
            knn = pickle.load(models)
        prediction_val = int(knn.predict(max_candle_cols))
    elif model == "XGBoost":
        with open((path + '/Models/{}/XGBoost_{}_{}.pickle'.format(model, symbol, interval)), 'rb') as models:
            xg_boost = pickle.load(models)
        prediction_val = int(xg_boost.predict(max_candle_cols))
        
    if prediction_val == 0:
        text = "Based on the {} model, the prediction for the next two {} intervals is DOWN. It is suggested that you sell this position.".format(model, interval)
    elif prediction_val == 1:
        text = "Based on the {} model, the prediction for the next two {} intervals is NEUTRAL. It is suggested that you take no action.".format(model, interval)
    elif prediction_val == 2:
        text = "Based on the {} model, the prediction for the next two {} intervals is UP. It is suggested that you take buy this product.".format(model, interval)
    
    return html.H3(text, style = title_style | text_center)


    
# %% show app

#if __name__ == '__main__': 
app.run_server()