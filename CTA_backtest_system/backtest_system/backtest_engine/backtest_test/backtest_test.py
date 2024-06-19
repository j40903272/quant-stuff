# -*- coding: utf-8 -*-
import webbrowser
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
import os
sys.path.insert(1, os.path.dirname(__file__) + '/../..')
user_home_path = str(Path.home())
import unittest
import configparser

from datetime import datetime

from backtest_engine.strategy_analyzer import StrategyAnalyzer
from backtest_engine.strategy_list import StrategyList

from dealer.px_handler import *
from dealer.configparser_dealer import *
from dealer.engine_data_handler import *

from data_loader.binance.binance_data_download import main as binance_data_download


"""
--strategy_logic SentimentAlgo  --strategy_config STA_BTC_1h_a01 --time_code all

--strategy_logic SmrcExpStrategy  --strategy_config SMRCEXP_ETH_15m_7772 --time_code R3M

--strategy_logic SmrcExpStrategy  --strategy_config SMRCEXP_ETH_15m_8934 --time_code 2022

--strategy_logic SqzCrossStrategy  --strategy_config SQZCROSS_ETH_60m_7802 --time_code selelct --start 2022-01-01 --end 2022-04-01
"""

def strategy_df_genator(config_path):

    today_date = datetime.today().strftime('%Y-%m-%d')

    config = configparser.ConfigParser()
    config.read(config_path)
    strategy_logic = config['basic']['strategy_logic']
    strategy_config = config['basic']['strategy_config']
    start_date = config['basic']['start_date']
    end_date = config['basic']['end_date']
    time_code = config['basic']['time_code']

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

    save_dir = f'./backtest_engine/backtest_test/{asset_class}'
    os.makedirs(save_dir + f"/{asset_class}", exist_ok=True)

    if asset_class == 'CRYPTO':
       data_df = px_handler(symbol, resample_p, asset_class, exchange)
    elif asset_class == 'FX':
       data_df = px_handler(symbol, resample_p, asset_class, exchange)
    else:
        raise(ValueError('Data Import Sth Wrong!!'))

    # "time_code" needs to preset.
    # "start_date" and "end_date" are valid only if "time_code" is set to "select".
    data_df = bar_slicer(time_code, data_df, today_date, start_date, end_date)

    # for printing
    strategy_select = StrategyList.get_strategy(strategy_logic=strategy_logic, strategy_config=f"{strategy_config}.json", data_df=data_df, start_date = start_date, end_date = end_date)
    
    cerebro_1 = StrategyAnalyzer(strategy_select)
    cerebro_1.backtest_analysis_report(save_result=True, print_result=True, show_plots=False, y_log_scale=True)

    metric_df_handler(asset_class, strategy_logic, strategy_config, data_df, save_dir, start_date, end_date)
    strategy_df_handler(asset_class, strategy_logic, strategy_config, data_df, save_dir, start_date, end_date)
    trade_pnl_handler(asset_class, strategy_logic, strategy_config, data_df, save_dir, start_date, end_date)

    # print("file://" + os.path.abspath(os.path.dirname(__file__)) + f'/../../backtest_result/html_report/{strategy_config}.html')
    # webbrowser.open("file://" + os.path.abspath(os.path.dirname(__file__)) + f'/../../backtest_engine/backtest_result/html_report/{strategy_config}.html')
    print(save_dir)
    return save_dir

class TestStrategy(unittest.TestCase):
    def test_strategy(self):
        save_dir = strategy_df_genator('./backtest_engine/backtest_test/TestList.ini')
        
        metric_df_dir = save_dir + f"/CRYPTO/ForceStrategy/metric_df/ForceStrategy_metric_df.csv"
        metric_df_test_dir = "./backtest_engine/backtest_test/test/ForceStrategy/metric_df/ForceStrategy_metric_df.csv"
        with open(metric_df_test_dir, 'r') as f1, open(metric_df_dir, 'r') as f2:
            self.assertListEqual(
                list(f1.read()),
                list(f2.read())
            )
        
        strategy_df_dir = save_dir + f"/CRYPTO/ForceStrategy/strategy_df/ForceStrategy_strategy_df.csv"
        strategy_df_test_dir = "./backtest_engine/backtest_test/test/ForceStrategy/strategy_df/ForceStrategy_strategy_df.csv"
        with open(strategy_df_test_dir, 'r') as f1, open(strategy_df_dir, 'r') as f2:
            self.assertListEqual(
                list(f1.read()),
                list(f2.read())
            )
        
        trade_pnl_df_dir = save_dir + f"/CRYPTO/ForceStrategy/trade_pnl_df/ForceStrategy_trade_pnl_df.csv"
        trade_pnl_df_test_dir = "./backtest_engine/backtest_test/test/ForceStrategy/trade_pnl_df/ForceStrategy_trade_pnl_df.csv"
        with open(trade_pnl_df_test_dir, 'r') as f1, open(trade_pnl_df_dir, 'r') as f2:
            self.assertListEqual(
                list(f1.read()),
                list(f2.read())
            )
        
if __name__ == '__main__':
    unittest.main()