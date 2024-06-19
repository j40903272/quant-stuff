import os
import sys
from pathlib import Path

sys.path.insert(1, os.path.dirname(__file__) + '/../../..')
user_home_path = str(Path.home())

import argparse
import configparser
import numpy as np
import pandas as pd
from tqdm import tqdm

from symbol_forex import *
from loading_center import *

"""
--platform NINJA_TRADER
--platform META_TRADER

"""

def fx_loader():

    fx_config = configparser.ConfigParser()
    fx_config.read(f'{user_home_path}/backtest_system/config/backtest_config.ini')
    meta_data_center_path = fx_config['forex_paths']['meta_trader']
    ninja_data_center_path = fx_config['forex_paths']['ninja_trader']

    parser = argparse.ArgumentParser(description='Forex Loader Option')
    parser.add_argument('--platform_code', help="platform", required=True)
    args = parser.parse_args()
    platform_code = args.platform_code

    if platform_code == 'NINJA_TRADER':
        platform_code == Platform.NINJA_TRADER
        output_directory = ninja_data_center_path

        for y in tqdm(loading_year):
            for p in fx_stmbol:
                output_filename, path_to_zip_file = download_hist_data(
                    year=y,
                    pair=p, 
                    month=None, 
                    platform=Platform.NINJA_TRADER, 
                    output_directory=output_directory, 
                    time_frame=TimeFrame.ONE_MINUTE
                )
                # unzip
                with zipfile.ZipFile(output_filename, 'r') as zip_ref:
                    zip_ref.extractall(path_to_zip_file)

    
    elif platform_code == 'META_TRADER':
        platform_code == Platform.META_TRADER
        output_directory = meta_data_center_path

        for y in tqdm(loading_year):
            for p in fx_stmbol:

                output_filename, path_to_zip_file = download_hist_data(
                    year=y, 
                    pair=p,
                    month=None, 
                    platform=Platform.META_TRADER, 
                    output_directory=output_directory, 
                    time_frame=TimeFrame.ONE_MINUTE
                )
                # unzip
                with zipfile.ZipFile(output_filename, 'r') as zip_ref:
                    zip_ref.extractall(path_to_zip_file)

if __name__ == '__main__':
    fx_loader()





# print(download_hist_data(year='2019', month='6', platform=Platform.NINJA_TRADER, time_frame=TimeFrame.ONE_MINUTE))
# print(download_hist_data(year='2019', month=None, platform=Platform.EXCEL, time_frame=TimeFrame.ONE_MINUTE))
# print(download_hist_data(year='2019', month='6', platform=Platform.META_STOCK, time_frame=TimeFrame.ONE_MINUTE))
# print(download_hist_data(year='2018', month=None, platform=Platform.NINJA_TRADER, time_frame=TimeFrame.ONE_MINUTE))
# print(download_hist_data(year='2018', month=None, platform=Platform.EXCEL, time_frame=TimeFrame.ONE_MINUTE))
# print(download_hist_data(year='2018', month=None, platform=Platform.META_TRADER, time_frame=TimeFrame.ONE_MINUTE))
# print(download_hist_data(year='2018', month=None, platform=Platform.META_STOCK, time_frame=TimeFrame.ONE_MINUTE))
