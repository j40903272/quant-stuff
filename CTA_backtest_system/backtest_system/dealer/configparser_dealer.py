import sys
from pathlib import Path
import os 

sys.path.insert(1, os.path.dirname(__file__) + '/../..')
user_home_path = str(Path.home())

import argparse
import configparser

def crpyto_dir_geneator():

    config = configparser.ConfigParser()
    config.read(f'./config/backtest_config.ini')
    spot_dir = config['paths']['spot_data']
    uperp_dir = config['paths']['uperp_data']

    return uperp_dir, spot_dir

def fx_dir_geneator():

    config = configparser.ConfigParser()
    config.read(f'{user_home_path}/backtest_system/config/backtest_config.ini')

    meta_trader_dir = config['forex_paths']['meta_trader_dir']
    ninja_trader_dir = config['forex_paths']['ninja_trader_dir']

    return meta_trader_dir, ninja_trader_dir