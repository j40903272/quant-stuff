import time
import os

import datetime
from datetime import datetime
import time


os.environ['TZ'] = 'Asia/Hong_Kong'
import pandas as pd 
import json
from pathlib import Path
import sys
user_home_path = str(Path.home())

sys.path.insert(1, os.path.dirname(__file__) + '/../../..')

import warnings
warnings.filterwarnings("ignore")

import requests
import configparser
config = configparser.ConfigParser()
config.read(f'{user_home_path}/backtest_system/config/confidential.ini')
key = config['td_ameritrade']['key']

def get_tda_quotes(**kwargs):

    url = 'https://api.tdameritrade.com/v1/marketdata/quotes'
    params = {}
    params.update({'apikey': key})
    symbol_list = []
    for symbol in kwargs.get('symbol'):
        symbol_list.append(symbol)
    params.update({'symbol': symbol_list})

    return requests.get(url, params=params).json()

def get_tda_historical_using_period(**kwargs):
    # this one using period 
    url = 'https://api.tdameritrade.com/v1/marketdata/{}/pricehistory'.format(kwargs.get('symbol'))
    params = {}
    params.update({
        'apikey': key,
        'periodType': 'day',
        'period':'10',
        'frequencyType': 'minute',
        'frequency': '1',
        'needExtendedHoursData': 'true',
        # 'startDate':'1640966400000',
        # 'endDate':'1663775940000',
        })

    result = requests.get(url, params=params).json()
    px = []
    for i in range(len(result['candles'])):
        px_dict = dict(
            datetime = result['candles'][i]['datetime'],
            open = result['candles'][i]['open'],
            high = result['candles'][i]['high'],
            low = result['candles'][i]['low'],
            close = result['candles'][i]['close'],
            volume = result['candles'][i]['volume'],
        )
        px.append(px_dict)
    px = pd.DataFrame(px)
    px['datetime'] = pd.to_datetime(px['datetime'], unit='ms')

    return px

def get_tda_historical_px_kwargs(**kwargs):
    # this one using million seconds not period
    url = 'https://api.tdameritrade.com/v1/marketdata/{}/pricehistory'.format(kwargs.get('symbol'))
    params = {}
    params.update({
        'apikey': key,
        'periodType': 'day',
        'frequencyType': 'minute',
        'frequency': '1',
        'needExtendedHoursData': 'true',
        'startDate':'{}'.format(kwargs.get('startDate')),
        'endDate':'{}'.format(kwargs.get('endDate')),
        })

    result = requests.get(url, params=params).json()
    px = []
    for i in range(len(result['candles'])):
        px_dict = dict(
            datetime = result['candles'][i]['datetime'],
            open = result['candles'][i]['open'],
            high = result['candles'][i]['high'],
            low = result['candles'][i]['low'],
            close = result['candles'][i]['close'],
            volume = result['candles'][i]['volume'],
        )
        px.append(px_dict)
    px = pd.DataFrame(px)
    px['datetime'] = pd.to_datetime(px['datetime'], unit='ms')

    return px

def get_tda_historical_px(symbol: str, start: int, end: int):
    
    # this one using symbol start and end, prepare for loop 
    url = f'https://api.tdameritrade.com/v1/marketdata/{symbol}/pricehistory'
    params = {}
    params.update({
        'apikey': key,
        'periodType': 'day',
        'frequencyType': 'minute',
        'frequency': '1',
        'needExtendedHoursData': 'true',
        'startDate':f'{start}',
        'endDate':f'{end}',
        })

    result = requests.get(url, params=params).json()
    px = []
    for i in range(len(result['candles'])):
        px_dict = dict(
            datetime = result['candles'][i]['datetime'],
            open = result['candles'][i]['open'],
            high = result['candles'][i]['high'],
            low = result['candles'][i]['low'],
            close = result['candles'][i]['close'],
            volume = result['candles'][i]['volume'],
        )
        px.append(px_dict)
    px = pd.DataFrame(px)
    px['datetime'] = pd.to_datetime(px['datetime'], unit='ms')

    return px

def anchor_list(current_update):

    delta = 3850000000 # the million second delta
    start = 946656000000 # 2000-01-01 00:00:00
    anchor = []
    times = (current_update-start)/delta
    for i in range(int(times+1)):
        current_update = current_update - delta
        anchor.append(current_update)
        
    return anchor

def test(symbol, start_1, end_1):
    # using million second test 
    url = f'https://api.tdameritrade.com/v1/marketdata/{symbol}/pricehistory'
    params = {}
    params.update({
        'apikey': key,
        'periodType': 'day',
        'frequencyType': 'minute',
        'frequency': '1',
        'needExtendedHoursData': 'true',
        'startDate':f'{start_1}',
        'endDate':f'{end_1}',
        })

    result = requests.get(url, params=params).json()
    print(result)
    px = []
    for i in range(len(result['candles'])):
        px_dict = dict(
            datetime = result['candles'][i]['datetime'],
            open = result['candles'][i]['open'],
            high = result['candles'][i]['high'],
            low = result['candles'][i]['low'],
            close = result['candles'][i]['close'],
            volume = result['candles'][i]['volume'],
        )
        px.append(px_dict)
    print(px)
    px = pd.DataFrame(px)
    px['datetime'] = pd.to_datetime(px['datetime'], unit='ms')

    return px