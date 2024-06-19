import sys
from pathlib import Path
from pprint import pprint
import hiplot as hip
from datetime import date, datetime
import pytz
import argparse
import configparser
import random
import os
import warnings
warnings.filterwarnings("ignore")

# locate the path 
sys.path.insert(1, os.path.dirname(__file__) + '/../..')
user_home_path = str(Path.home())

import numpy as np
import pandas as pd

from backtest_engine.strategy_list import StrategyList
from backtest_engine.strategy_analyzer import StrategyAnalyzer
from backtest_engine.exchange_enum import StrategySettings
from backtest_engine.utils import *
from dealer.configparser_dealer import crpyto_dir_geneator

"""
--strategy_logic SmrcExpStrategy --symbol BTC --tuning_num 50 --candidate_strategy SMRCEXP_LINK_30m_8731

"""

random_search_dir = f"{user_home_path}/backtest_system/backtest_engine/strategy_tuner/tuning_result"

# bar path
config = configparser.ConfigParser()
config.read(f"./config/backtest_config.ini")

uperp_data = config["paths"]["uperp_data"]
spot_data = config["paths"]["spot_data"]


def random_tuner():

    tuning_parser = argparse.ArgumentParser(description='random tuning')
    tuning_parser.add_argument('--symbol', help="symbol", required=True) # BTC ETH BNB
    tuning_parser.add_argument('--strategy_logic', help="strategy class", required=True)
    tuning_parser.add_argument('--training_pct', help="training pct", default=0.7)
    tuning_parser.add_argument('--tuning_num', help="tuning num", required=True)
    tuning_parser.add_argument('--candidate_strategy', help="tuning json", required=True)

    args = tuning_parser.parse_args()
    symbol = str(args.symbol)
    strategy_logic = args.strategy_logic
    candidate_strategy = args.candidate_strategy
    training_pct = float(args.training_pct)
    tuning_num = int(args.tuning_num)
    spot_end_date = '2020-01-01'

    file_ref = random.randint(1001, 10000)
    print("-------------------------")
    print("<Backtest Engine Alpha>")
    print("-------------------------")
    print("Plateau Searching Running")
    print("-------------------------")
    print(f"Tuning Asset: {symbol} | Strategy: {strategy_logic}")
    print("-------------------------")
    print(f"Tuning Json: {candidate_strategy} | Experiment Reference: {file_ref}")
    print("-------------------------")
    
    tuning_data_dir = f"{user_home_path}/backtest_system/backtest_engine/search_result/random_search/{candidate_strategy}_plateau_search_rs{file_ref}.csv"

    SPOT_TICKER = symbol + "USDT_SPOT"
    spot_data_df = pd.read_csv(spot_data + f"/SPOT/minute/{SPOT_TICKER}_1m.csv", index_col="datetime")
    spot_data_df.index = pd.to_datetime(spot_data_df.index, format="%Y-%m-%d %H:%M:%S")
    spot_df = spot_data_df.loc[:f'{spot_end_date}']

    UPERP_TICKER = symbol + "USDT_UPERP"
    uperp_data_df = pd.read_csv(uperp_data + f"/UPERP/minute/{UPERP_TICKER}_1m.csv", index_col="datetime")
    uperp_data_df.index = pd.to_datetime(uperp_data_df.index, format="%Y-%m-%d %H:%M:%S")

    # training pct
    train_df = uperp_data_df[:(int(len(uperp_data_df) * training_pct))]
    test_df = uperp_data_df[(int(len(uperp_data_df) * training_pct)):]

    tune_strategy = StrategyList.get_strategy(strategy_logic, f"{candidate_strategy}_plateau_search.json", train_df, plateau=True)

    pf_analyzer = StrategyAnalyzer(tune_strategy, train_df)
    pf_analyzer.random_search_with_validation(train_data=train_df, test_data=test_df, spot_data=spot_df, n_iter=tuning_num, random_state=file_ref)
    tuning_data = pd.read_csv(tuning_data_dir)

    # to local
    local_path = f"{user_home_path}/backtest_system/backtest_engine/strategy_tuner/tuning_result/{strategy_logic}"
    os.makedirs(local_path, exist_ok=True)

    tuning_data.to_csv(local_path + f"/PLATEAU_{strategy_logic}_{symbol}_rs{file_ref}.csv")
    print(f"<{strategy_logic}_{file_ref}> tuning result saved in local file: " f"{local_path}" + f"/PLATEAU_{strategy_logic}_{symbol}_rs{file_ref}.csv!")

    print("-------------------------")
    print("<Backtest Engine Alpha>")
    print("-------------------------")
    print("Plateau Searching Completed")
    print("-------------------------")
    print(f"Tuning Asset: {symbol} | Strategy: {strategy_logic}")
    print("-------------------------")
    print(f"Tuning Json: {candidate_strategy} | Experiment Reference: {file_ref}")
    print("-------------------------")
    print("<Backtest Engine Alpha>")
    print("-------------------------")

if __name__ == '__main__':
    random_tuner()
