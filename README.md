# ML_Tradebot
Binance Testnet Tradebot with Machine Learning

This Github contains the code for our machine-learning tradebot. Our final solution is built in Python Dash and is fully interactive, however, due to Binance.com API rules it does require a VPN to run (this rule was implemented by Binance during the course of our capstone projet).

The folder structure inside has been documented below. 

- Main:
  - Dash Test: Contains many scripts and artefacts that were developed as small-scale tests for our solution. Can be ignored.
  - Production:
    (Files)
    - Robot_logo.png: image used in Dash App
    - config.txt: contains API key/secret information for Binance.com Testnet API.
    - binance_data_refresh.py: custom script with functions to download data from Binance API and enrich with trading indicators. Called via Dash app.
    - model_retrain.py: custom script for re-training models and storing as pickle files. Called via Dash app.
    - DashApp_101.py: The main script for this entire solution. Running this script will activate the Dash app. 
    
    (Folders)
    - *Datasets:* Folder containing all raw, interim and final datasets used in the solution.
    - Models: Folder containing model pickle files which are updated and utilized in the solution.
    - Weekly Notebooks: Ad-hoc notebooks that were used to translate Dash app componenets for the final paper (e.g. analysis on data and models used in the                         app)
