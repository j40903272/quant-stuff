import os
import sys
from pathlib import Path

sys.path.insert(1, os.path.dirname(__file__) + '/../../..')
user_home_path = str(Path.home())

import argparse
import configparser
import numpy as np
import pandas as pd
from symbol_forex import *
from tqdm import tqdm
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

fx_config = configparser.ConfigParser()
fx_config.read(f'{user_home_path}/backtest_system/config/backtest_config.ini')
meta_data_original_path = fx_config['forex_paths']['meta_trader_original']
ninja_data_original_path = fx_config['forex_paths']['ninja_trader_original']
fx_data_center_path = fx_config['forex_paths']['fx_data_center_path']

category = ['GBPUSD','EURUSD','USDJPY'] # this can be import future

def metatrader_fx_agg():

    metatrader_fx_df = pd.DataFrame()
    for i in tqdm(category):
        for j in tqdm(loading_year) :
            MT = pd.read_csv(f'{meta_data_original_path}/FX_MT_{str(i)}_M1_{int(j)}.csv/DAT_MT_{str(i)}_M1_{int(j)}.csv',header=None)
            MT.columns = ['d','t','open','high','low','close','v']
            MT['datetime'] = MT['d'].astype(str) + ' ' + MT['t']
            MT = MT.drop(['d','t','v'],axis=1)
            MT = MT[['datetime','open','high','low','close']]
            MT['datetime'] = pd.to_datetime(MT['datetime'], format='%Y-%m-%d %H:%M:%S')
            metatrader_fx_df = metatrader_fx_df.append(MT)
        print('--------------')    
        print(metatrader_fx_df)
        print('--------------') 
        os.makedirs(fx_data_center_path, exist_ok=True)
        os.makedirs(fx_data_center_path + '/meta_trader', exist_ok=True)
        metatrader_fx_df.to_csv(fx_data_center_path + f'/meta_trader/{str(i)}_1m.csv')
        print(f'FX data saved in fx_data_center_path' + f'/meta_trader/{str(i)}_1m.csv')
    print('meta_trader aggregation done!')

def ninja_fx_df_agg():
    
    ninjatrader_fx_df = pd.DataFrame()
    for i in tqdm(category):
        for j in tqdm(loading_year):
            NT = pd.read_csv(f'{ninja_data_original_path}/FX_NT_{str(i)}_M1_{int(j)}.csv/DAT_NT_{i}_M1_{int(j)}.csv',header=None)
            NT = NT[0].str.split(';',n=5,expand=True)
            NT[0] = pd.to_datetime(NT[0], format='%Y-%m-%d %H:%M:%S')
            NT = NT.drop([5],axis=1)
            NT.columns = [['datetime','open','high','low','close']]
            ninjatrader_fx_df = ninjatrader_fx_df.append(NT)
        print('--------------')    
        print(ninjatrader_fx_df)
        print('--------------') 
        os.makedirs(fx_data_center_path, exist_ok=True)
        os.makedirs(fx_data_center_path + '/ninja_trader', exist_ok=True)
        ninjatrader_fx_df.to_csv(fx_data_center_path + f'/ninja_trader/{str(i)}_1m.csv')
        print(f'FX data saved in fx_data_center_path' + f'/ninja_trader/{str(i)}_1m.csv')
    print('ninja_trader aggregation Done!')


if __name__ == '__main__':
    metatrader_fx_agg()
    ninja_fx_df_agg()