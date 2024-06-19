from pathlib import Path
import sys
import os

sys.path.insert(1, os.path.dirname(__file__) + '/../..')
user_home_path = str(Path.home())

import os
import pandas as pd
import argparse
import configparser
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from dealer.configparser_dealer import *


def spot_data_import(symbol, spot_dir):

    min_data = pd.read_csv(spot_dir + f'SPOT/minute/{symbol}_1m.csv', index_col='datetime')
    min_data.index = pd.to_datetime(min_data.index, format='%Y-%m-%d %H:%M:%S')

    return min_data

def uperp_data_import(symbol, uperp_dir):

    # UPERP/SPOT data import
    min_data = pd.read_csv(uperp_dir + f'UPERP/minute/{symbol}_1m.csv', index_col='datetime')
    min_data.index = pd.to_datetime(min_data.index, format='%Y-%m-%d %H:%M:%S')

    return min_data

def uperp_data_import_kfold(symbol, uperp_dir):

    # UPERP/SPOT data import 
    min_data = pd.read_csv(uperp_dir + f'UPERP/minute/{symbol}USDT_UPERP_1m.csv', index_col='datetime')
    min_data.index = pd.to_datetime(min_data.index, format='%Y-%m-%d %H:%M:%S')

    return min_data

def fx_data_import(symbol, exchange):

    meta_trader_dir, ninja_trader_dir = fx_dir_geneator()

    if exchange == 'META':
        fx_min_data = pd.read_csv(meta_trader_dir + f'/{symbol}_1m.csv', index_col='datetime')
        fx_min_data.index = pd.to_datetime(fx_min_data.index, format='%Y-%m-%d %H:%M:%S')
    elif exchange == 'NIJA':
        fx_min_data = pd.read_csv(ninja_trader_dir + f'/{symbol}_1m.csv', index_col='datetime')
        fx_min_data.index = pd.to_datetime(fx_min_data.index, format='%Y-%m-%d %H:%M:%S')

    return fx_min_data

def px_handler(symbol, resample_p, asset_class, exchange, randomness_std):

    if asset_class == "CRYPTO":
        uperp_dir, _ = crpyto_dir_geneator()
        min_data = uperp_data_import(symbol, uperp_dir)
        
        if randomness_std:
            # 将'close'列乘以正态分布生成的随机数
            min_data['close'] = min_data['close'].mul(np.random.normal(1, randomness_std, len(min_data.index)))
            # 将修改后的'close'列的值赋给'open'列的除第一个元素外的所有元素
            min_data['open'].iloc[1:] = min_data['close'].iloc[:-1]
            
        data_df = resample_data(min_data, resample_p)

    elif asset_class == "FX":
        fx_min_data = fx_data_import(symbol, exchange)
        data_df = resample_data(fx_min_data, resample_p) 

    return data_df

def mondo_px_handler(db, collection, resample_p,  query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read from Mongo and Store into DataFrame """
    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)
    # Expand the cursor and construct the DataFrame
    min_data =  pd.DataFrame(list(cursor))
    # Delete the _id
    if no_id and '_id' in min_data:
        del min_data['_id']
    data_df = resample_data(min_data, resample_p, end_bar_time=False)

    return min_data

def resample_data(min_data, resample_p, end_bar_time=False):
    data_df = pd.DataFrame()
    for col in min_data.columns:
        if 'open' in col:
            data_df[col] = min_data[col].resample(resample_p, origin='epoch').first()
        if 'high' in min_data.columns:
            data_df['high'] = min_data['high'].resample(resample_p, origin='epoch').max()
        if 'low' in min_data.columns:
            data_df['low'] = min_data['low'].resample(resample_p, origin='epoch').min()
        if 'close' in col:
            data_df[col] = min_data[col].resample(resample_p, origin='epoch').last()
        if 'volume' in min_data.columns:
            data_df['volume'] = min_data['volume'].resample(resample_p, origin='epoch').sum()
    if end_bar_time:
        data_df = data_df.shift(1)
    data_df = data_df.dropna()

    return data_df

def bar_slicer(time_code, data_df, today_date, start_date: None, end_date: None): # No longer used
    
    if time_code == 'all':
        selected_data_df = data_df.loc[:today_date]
    elif time_code == 'select':
        selected_data_df = data_df.loc[f'{start_date} 18:30:00':f'{end_date} 18:30:00']
    # year
    elif time_code == '2020':
        selected_data_df = data_df.loc['2020-01-01 18:30:00':'2021-01-01 18:30:00']
    elif time_code == '2021':
        selected_data_df = data_df.loc['2021-01-01 18:30:00':'2022-01-01 18:30:00']
    elif time_code == '2022':
        selected_data_df = data_df.loc['2022-01-01 18:30:00':f'{today_date} 18:30:00']
    # half year
    elif time_code == '1H20':
        selected_data_df = data_df.loc['2020-01-01 18:30:00':'2020-07-01 18:30:00']
    elif time_code == '2H20':
        selected_data_df = data_df.loc['2020-07-01 18:30:00':'2021-01-01 18:30:00']
    elif time_code == '1H21':
        selected_data_df = data_df.loc['2021-01-01 18:30:00':'2021-07-01 18:30:00']
    elif time_code == '2H21':
        selected_data_df = data_df.loc['2021-07-01 18:30:00':'2022-01-01 18:30:00']
    # quarter
    elif time_code == '1Q20':
        selected_data_df = data_df.loc['2020-01-01 18:30:00':'2020-04-01 18:30:00']
    elif time_code == '2Q20':
        selected_data_df = data_df.loc['2020-04-01 18:30:00':'2020-07-01 18:30:00']
    elif time_code == '3Q20':
        selected_data_df = data_df.loc['2020-07-01 18:30:00':'2020-10-01 18:30:00']
    elif time_code == '4Q20':
        selected_data_df = data_df.loc['2020-10-01 18:30:00':'2021-01-01 18:30:00']
    elif time_code == '1Q21':
        selected_data_df = data_df.loc['2021-01-01 18:30:00':'2021-04-01 18:30:00']
    elif time_code == '2Q21':
        selected_data_df = data_df.loc['2021-04-01 18:30:00':'2021-07-01 18:30:00']
    elif time_code == '3Q21':
        selected_data_df = data_df.loc['2021-07-01 18:30:00':'2021-10-01 18:30:00']
    elif time_code == '4Q21':
        selected_data_df = data_df.loc['2021-10-01 18:30:00':'2022-01-01 18:30:00']
    elif time_code == '1Q22':
        selected_data_df = data_df.loc['2022-01-01 18:30:00':'2022-04-01 18:30:00']
    elif time_code == '2Q22':
        selected_data_df = data_df.loc['2022-04-01 18:30:00':'2022-07-01 18:30:00']
    elif time_code == '3Q22':
        selected_data_df = data_df.loc['2022-07-01 18:30:00':f'2022-10-01 18:30:00']
    elif time_code == '4Q22':
        selected_data_df = data_df.loc['2022-10-01 18:30:00':f'{today_date} 18:30:00']
    elif time_code == '2H22':
        selected_data_df = data_df.loc['2022-07-01 18:30:00':f'{today_date} 18:30:00']
    # for special request
    elif time_code == 'R1M':
        selected_data_df = data_df.loc['2022-10-01 18:30:00':f'{today_date} 18:30:00']
    elif time_code == 'R3M':
        selected_data_df = data_df.loc['2022-09-01 18:30:00':f'{today_date} 18:30:00']
    elif time_code == 'test':
        #selected_data_df = data_df.loc['2023-04-01 00:00:00': f'{(datetime.utcnow() - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:00")}']
        selected_data_df = data_df.loc['2023-04-01 18:30:00':'2023-06-13 18:30:00']
    else:
        raise ValueError('Wrong Period!')

    return selected_data_df

def ohlc_turn_cap(data: pd.DataFrame):
    # for backtesting
    data.rename(columns = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace = True)
    return data

def get_timeframe_string(portfolio_df):

    timeframe = portfolio_df.index.to_series().diff()[-1]
    if timeframe == timedelta(minutes=5):
        timeframe_in_string = '5m'
    elif timeframe == timedelta(minutes=15):
        timeframe_in_string = '15m'
    elif timeframe == timedelta(minutes=30):
        timeframe_in_string = '30m'
    elif timeframe == timedelta(hours=1):
        timeframe_in_string = '1h'
    elif timeframe == timedelta(hours=2):
        timeframe_in_string = '2h'
    elif timeframe == timedelta(hours=4):
        timeframe_in_string = '4h'
    elif timeframe == timedelta(hours=8):
        timeframe_in_string = '8h'
    elif timeframe == timedelta(hours=12):
        timeframe_in_string = '12h'
    elif timeframe == timedelta(days=1):
        timeframe_in_string = '1d'
    elif timeframe == timedelta(days=2):
        timeframe_in_string = '2d'
    elif timeframe == timedelta(days=3):
        timeframe_in_string = '3d'

    return timeframe_in_string

def dict_without_keys(dict, keys):
   return {x: dict[x] for x in dict if x not in keys}



    
