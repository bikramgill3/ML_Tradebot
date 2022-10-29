#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 13:47:44 2022

@author: bikramgill
"""

# %% import libraries

import dash
from dash import html, dcc
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

#from testfunction import output_csv

# %% Notes and To-Dos

# 1) Change color palette and use the color codes you found on instagram


# %% set OS filepath

path = os.path.dirname(__file__) 

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
    'color': 'white',
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
                    dcc.Dropdown(['XGBoost', 'Logistic Regression'], 'XGBoost', id='model_type', clearable=False)
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
                html.Div([html.Button('Re-Train Model', id='model_train', n_clicks=0)], style=button_style)
                
                
                
                ], style={'display': 'inline-block', 'width':'50%'}),
        
            ## Model Div 1.2: Model Information and errors. 
            html.Div([
                html.H3('Model Performance', style = title_style | text_center),
                ],style={'display': 'inline-block', 'width':'50%'}),
            
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
                     dcc.Tab(label='Trading', value='tab_trading', style=tab_style, selected_style=tab_selected_style),
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
    binance_data.add_features()
    binance_data.output_df()
    
    graph_df = pd.read_csv(os.path.join(path, 'Datasets/', symbol + '_' + interval + '.csv'), parse_dates=['close_time'])
        
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
# @app.callback(
#     Input('model_train', 'n_clicks')
#     State('model_type','value')
#     State('model_market_type','value')
#     State('model_interval','value')
# )

    
# %% show app

#if __name__ == '__main__': 
app.run_server()