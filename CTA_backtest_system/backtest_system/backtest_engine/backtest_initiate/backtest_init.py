# -*- coding: utf-8 -*-
import webbrowser
import json
import sys
from pathlib import Path
import os
sys.path.insert(1, os.path.dirname(__file__) + '/../..')
user_home_path = str(Path.home())

import argparse
from datetime import datetime

from dealer.px_handler import *
from dealer.configparser_dealer import *
from dealer.engine_data_handler import *

from data_loader.binance.binance_data_download import main as binance_data_download

def main():

    today_date = datetime.today().strftime('%Y-%m-%d')
    parser = argparse.ArgumentParser(description='strategy backtest')
    parser.add_argument('--strategy_logic', help="strategy logic", required=True)
    parser.add_argument('--strategy_config', help="strategy config", required=True)
    parser.add_argument('--start', help="start time", required=True)
    parser.add_argument('--end', help="end time", required=True)
    parser.add_argument('--random_std', help="randomness standard deviation", type=float, required=False)

    args = parser.parse_args()
    strategy_logic = args.strategy_logic
    strategy_config = args.strategy_config
    start_date = args.start
    end_date = args.end
    random_std = args.random_std

    # read the engine json
    with open(f'./backtest_engine/strategies_parameters/{strategy_logic}/{strategy_config}.json') as f:
        config = json.loads(f.read())
    
    asset_class = 'CRYPTO'
    resample_p = config['prepare_data_param']['resample']
    exchange = config['exchange']
    symbol = config['symbol']

    # ETHUSDT_UPERP -> ETHUSDT UPERP
    sub_symbol = symbol.split('_')
    # Download data before backtesting
    binance_data_download(sub_symbol[0],'1m',sub_symbol[1])

    save_dir = f'./backtest_engine/data_center/{asset_class}'
    os.makedirs(save_dir + f"/{asset_class}", exist_ok=True)

    if asset_class == 'CRYPTO':
       data_df = px_handler(symbol, resample_p, asset_class, exchange, random_std)
    elif asset_class == 'FX':
       data_df = px_handler(symbol, resample_p, asset_class, exchange)
    else:
        raise(ValueError('Data Import Sth Wrong!!'))

    # "time_code" needs to preset.
    # "start_date" and "end_date" are valid only if when "time_code" is set to "select".
    # data_df = bar_slicer(time_code, data_df, today_date, start_date, end_date)

    cerebro = cerebro_init(strategy_logic, strategy_config, data_df, start_date, end_date)
    generate_backtest_result_csv(cerebro, asset_class, strategy_logic, strategy_config, save_dir)

    print("file://" + os.path.abspath(os.path.dirname(__file__)) + f'/../../backtest_result/html_report/{strategy_config}.html')
    webbrowser.open("file://" + os.path.abspath(os.path.dirname(__file__)) + f'/../../backtest_engine/backtest_result/html_report/{strategy_config}.html')
    
if __name__ == '__main__':
    main()









