import argparse
import concurrent.futures
import configparser
import logging
import os
import random
import time
from datetime import datetime, timedelta
import ccxt
import pandas as pd
from ccxt import ExchangeError

import sys
from pathlib import Path

sys.path.insert(1, os.path.dirname(__file__) + '/../../..')

user_home_path = str(Path.home())


from data_loader.csv_dealer import CsvDealer

FORMAT = '%Y-%m-%d %H:%M:%S'


def timestamp_to_int(dt):
    """
    takes a datetime object
    return timestamp in int
    """
    return int(datetime.timestamp(dt) * 1000)

def timestampt_to_datetime(ts):
    """
    takes a timestamp in int
    return datetime object
    """
    return datetime.utcfromtimestamp(ts / 1000)

def paginate(client, symbol, timeframe, directory, type_, start_dt, to_dt=datetime.utcnow()):
    data = []
    file = f'{directory}/{symbol.replace("/", "")}_{type_}_{timeframe}.csv'
    if os.path.isfile(file):
        csv_writer = CsvDealer(file, None, None, FORMAT)
        start_dt = csv_writer.get_last_row_date_csv(format=FORMAT)
    since = timestamp_to_int(start_dt)
    while True:
        try:
            patch = client.fetch_ohlcv(symbol, timeframe, since)
            data += patch
            if patch:
                print(f'[{symbol}] Fetched data from - {patch[0][0]} - {timestampt_to_datetime(patch[0][0])} to {timestampt_to_datetime(patch[-1][0])}')
                # update next patch to have since = last ts of patch + 1 millisec to avoid duplication of data
                since = int(patch[-1][0]) + 1
                if timestampt_to_datetime(since) > to_dt - timedelta(hours=1):
                    break
                else:
                    time.sleep(client.rateLimit / 1000)
            else:
                if timeframe == '1h':
                    since += 2592000000  # 1 month
                else:
                    since += 60000 * 1000
                    # since += 2592000000  # 1 month
                print(f'[{symbol}] Trying later start dt - {timestampt_to_datetime(since)}')
                if timestampt_to_datetime(since) > datetime.utcnow():
                    break
        except ExchangeError as e:
            print(f'[{symbol}] Getting error {e}')
            break

    print(f'Acquired # of timepoints: {len(data)}')
    if len(data) == 0:
        return

    print(f'First data point: {data[0]}')
    print(f'Last data point: {data[-1]}')

    # last bar is incomplete
    data = data[:-1]
    if os.path.isfile(file):
        for row in data:
            csv_writer.write_if_needed([datetime.utcfromtimestamp(row[0] / 1000), row[1], row[2], row[3], row[4], row[5]], start_dt)
    else:
        df = pd.DataFrame({
            'datetime': [datetime.utcfromtimestamp(row[0] / 1000).strftime(FORMAT) for row in data],
            'open': [row[1] for row in data],
            'high': [row[2] for row in data],
            'low': [row[3] for row in data],
            'close': [row[4] for row in data],
            'volume': [row[5] for row in data]
        })

        df.to_csv(file, index=False)
    time.sleep(1)

def get_start_dt(symbols_detail_map, symbol):
    # print(symbols_detail_map)
    if symbol in symbols_detail_map and 'onboardDate' in symbols_detail_map[symbol]['info']:
        symbol_onboard_date = datetime.fromtimestamp(int(symbols_detail_map[symbol]['info']['onboardDate']) / 1000 - 1000)
        start_dt = symbol_onboard_date if symbol_onboard_date > SINCE else SINCE
    else:
        start_dt = SINCE
    return start_dt