import os
import sys
import time
import pytz
from datetime import datetime
import argparse
import configparser
from pathlib import Path

sys.path.insert(1, os.path.dirname(__file__) + '/../..')

user_home_path = str(Path.home())

import ccxt
from data_loader.csv_dealer import CsvDealer
from data_loader.loader_utils import *

"""

SPOT
-s all_spot -t 1m --type SPOT

UPERP
-s all -t 1m --type UPERP

CPERP
-s all -t 1m --type CPERP

"""

def main(symbol, timeframe, type):

    SINCE = datetime(2016, 1, 1, tzinfo=pytz.timezone('UTC'))
    TO = datetime.utcnow()

    config = configparser.ConfigParser()
    config.read(f'./config/backtest_config.ini')

    # for one drive bar data path 
    data_center_path = config['paths']['uperp_data']
    print(data_center_path)
    os.makedirs(data_center_path, exist_ok=True)

    if type == 'SPOT':
        data_path = f'{data_center_path}/SPOT'
    elif type == 'UPERP':
        data_path = f'{data_center_path}/UPERP'
    elif type == 'CPERP':
        data_path = f'{data_center_path}/CPERP'

    if timeframe == '1m':
        data_path = data_path + '/minute'
    elif timeframe == '1h':
        data_path = data_path + '/hour'
    elif timeframe == '4h':
        data_path = data_path + '/4h'
    elif timeframe == '1d':
        data_path = data_path + '/daily'

    os.makedirs(data_path, exist_ok=True)

    if type == 'SPOT':
        client = ccxt.binance()
    elif type == 'UPERP':
        client = ccxt.binanceusdm()
    elif type == 'CPERP':
        client = ccxt.binancecoinm()

    if symbol == 'all':
        symbol_details = client.fetch_markets()
        for i in symbol_details:
            symbol_ = i['symbol']
            symbol_onboard_date = datetime.fromtimestamp(int(i['info']['onboardDate']) / 1000 - 1000)
            start_dt = symbol_onboard_date if symbol_onboard_date > SINCE else SINCE
            print(f'Getting data: {start_dt}, {symbol_}, {timeframe}')
            paginate(client, symbol_, timeframe, data_path, type, start_dt, TO)

    else:
        symbol_details = client.fetch_markets()
        symbols = [i['symbol'] for i in symbol_details]
        if symbol in symbols:
            symbol_onboard_date = datetime.fromtimestamp(int(symbol_details[0]['info']['onboardDate']) / 1000 - 1000)
            start_dt = symbol_onboard_date if symbol_onboard_date > SINCE else SINCE
        else:
            start_dt = SINCE

        print(f'Getting data: {start_dt}, {symbol}, {timeframe}')
        paginate(client, symbol, timeframe, data_path, type, start_dt, TO)
    print('Saved data.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='symbol and timeframe')
    parser.add_argument('-s', help="Symbol.")
    parser.add_argument('-t', help="Timeframe.")
    parser.add_argument("--type", "--type", action="store", dest="type", help="SPOT USD COIN.")
    args = parser.parse_args()

    main(args.s, args.t, args.type)
