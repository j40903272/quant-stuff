import os
import sys
from pathlib import Path
sys.path.insert(1, os.path.dirname(__file__) + '/../..')
import numpy as np
import pandas as pd
import configparser
import argparse
import random
import json

from backtest_engine.strategy_analyzer import StrategyAnalyzer
from backtest_engine.strategy_list import StrategyList

from params_selector import *
from dealer.px_handler import *

config = configparser.ConfigParser()
config.read(f'./backtest_system/config/backtest_config.ini')
uperp_dir = config['paths']['uperp_data']

"""
--strategy_class SqzCrossExpandStrategy --testing_symbol ETH --k_folds 5 --n_iter 300

--strategy_class SmrcExpStrategy --testing_symbol ETH --k_folds 5 --n_iter 100
"""

def main():

    k_parser = argparse.ArgumentParser(description='k fold tuning')
    k_parser.add_argument('--strategy_class', help="strategy_class", required=True)
    k_parser.add_argument('--k_folds', help="k_folds", required=True)
    k_parser.add_argument('--n_iter', help="n_iter", required=True)
    k_parser.add_argument('--valid_pct', help="valid_pct", default=0.1)
    k_parser.add_argument('--train_pct', help="train_pct", default=0.7)
    k_parser.add_argument('--dump_json', help="whether dump json file", default=False)
    k_parser.add_argument('--fold_sep', help="fold_sep", default=False)
    k_parser.add_argument('--testing_symbol', help="testing_symbol", required=True)
    k_parser.add_argument('--save_drive', help="save_drive", default=True)

    args = k_parser.parse_args()
    strategy_class = args.strategy_class
    K_FOLDS = int(args.k_folds)
    K_FOLDS_PCT = 1 / K_FOLDS
    n_iter = int(args.n_iter)
    VALID_PCT = float(args.valid_pct)
    TRAIN_PCT = float(args.train_pct)
    dump_json = args.dump_json
    fold_sep = args.fold_sep
    save_drive = args.save_drive
    testing_symbol = args.testing_symbol

    with open(f'{user_home_path}/backtest_system/backtest_engine/strategies_parameters/{strategy_class}/{strategy_class}_{testing_symbol}_kf.json') as f:
        data = json.load(f)
        symbol = data['symbol'].replace("USDT_UPERP", "")  # input shd be this - ETHUSDT_UPERP, we want symbol to be ETH for read_data_csv

    uperp_data_df = uperp_data_import_kfold(symbol, uperp_dir)

    valid_data = uperp_data_df[int(len(uperp_data_df) * (1 - VALID_PCT)):]
    all_train_data = uperp_data_df[:int(len(uperp_data_df) * (1 - VALID_PCT))]
    tune_train_df = []
    tune_test_df = []

    for i in range(K_FOLDS + 1):

        if fold_sep:
            tune_df = (all_train_data[int(len(all_train_data) * K_FOLDS_PCT * (i)):int(len(all_train_data) * K_FOLDS_PCT * (i + 1))])
        else:
            tune_df = (all_train_data[:int(len(all_train_data) * K_FOLDS_PCT * (i + 1))])
        tune_train_df.append(tune_df[:int(len(tune_df) * TRAIN_PCT)])
        tune_test_df.append(tune_df[int(len(tune_df) * TRAIN_PCT):])

    file_ref = random.randint(1001, 10000)
    print(f"K Fold Tuning Running <K={K_FOLDS}>...")
    print("|------------------------------------|")
    print(f"symbol: {symbol} | strategy_class: {strategy_class}")
    print("|------------------------------------|")
    print(f"Reference num start with: {file_ref}")
    print("|------------------------------------|")
    print(f"Final Reference : {file_ref + K_FOLDS }")

    train_list = []
    test_list = []
    valid_list = []
    params_list = []
    datetime_list = []
    fold_record_list = []

    for i in range(1, K_FOLDS + 1):
        strategy = StrategyList.get_strategy(f'{strategy_class}', f'{strategy_class}_{testing_symbol}_kf.json', tune_train_df[i-1])
        pf_analyzer = StrategyAnalyzer(strategy)
        file_ref = file_ref + 1

        train_start = tune_train_df[i - 1].index[0]
        train_end = tune_train_df[i - 1].index[-1]
        test_start = tune_test_df[i - 1].index[0]
        test_end = tune_test_df[i - 1].index[-1]
        valid_data_start = valid_data.index[0]
        valid_data_end = valid_data.index[-1]
        df = [i, train_start, train_end, test_start, test_end, valid_data_start, valid_data_end]
        datetime_df = pd.DataFrame(df)
        datetime_df = datetime_df.transpose()
        datetime_df.columns = ['fold', 'train_start', 'train_end', 'test_start', 'test_end', 'valid_data_start', 'valid_data_end']

        # for adding the final result
        fold_record = [i]
        fold_record_df = pd.DataFrame(fold_record)
        fold_record_df = fold_record_df.transpose()
        fold_record_df.columns = ['fold']

        print("|------------------------------------|")
        print(f"K Fold in {i}")
        print(f"Search Period: {train_start} to {train_end}")
        print(f"Test Period: {test_start} to {test_end}")
        print("|-----------train--------------------|")
        print("head")
        print(tune_train_df[i-1].head(3))
        print("tail")
        print(tune_train_df[i-1].tail(3))
        print("|-----------test---------------------|")
        print("head")
        print(tune_test_df[i-1].head(5))
        print("tail")
        print(tune_test_df[i-1].tail(5))
        print("|------------------------------------|")

        # tuning
        pf_analyzer.random_search_with_validation(
            train_data=tune_train_df[i - 1],
            test_data=tune_test_df[i - 1],
            n_iter=n_iter,
            random_state=file_ref,
            spot_data=valid_data
        )

        file_path = f'{user_home_path}/backtest_system/backtest_engine/search_result/random_search/{strategy_class}_{testing_symbol}_kf_rs{file_ref}.csv'

        params = param_set_determine(file_path)
        selected_data = params[:'no_of_layer']
        print(selected_data)
        prepare_data_selected = params[:'SL_mode']
        trading_data_selected = params['SL_mode':'no_of_layer']
        params_selected = params[:'no_of_layer'].to_frame()
        stats_data_train = params['sharpe_train':'total_win_rate_train'].to_frame()
        stats_data_test = params['sharpe_test':'total_win_rate_test'].to_frame()
        stats_data_valid = params['sharpe_spot':'total_win_rate_spot'].to_frame()

        stats_data_train = stats_data_train.transpose()
        stats_data_test = stats_data_test.transpose()
        stats_data_valid = stats_data_valid.transpose()
        params_selected = params_selected.transpose()

        stats_data_valid.rename(columns={
            'sharpe_spot': 'sharpe_valid',
            'annualized_returns_spot': 'annualized_returns_valid',
            'trade_per_day_spot': 'trade_per_day_valid',
            'max_drawdown_spot': 'max_drawdown_valid',
            'ann_return/MDD_spot': 'ann_return/MDD_valid',
            'annualized_returns_spot': 'annualized_returns_valid',
            'total_win_rate_spot': 'total_win_rate_valid'
        }, inplace=True)

        train_list.append(stats_data_train)
        test_list.append(stats_data_test)
        valid_list.append(stats_data_valid)
        params_list.append(params_selected)
        datetime_list.append(datetime_df)
        fold_record_list.append(fold_record_df)

        data['prepare_data_param'] = prepare_data_selected.to_dict()
        data['trading_data_param'] = trading_data_selected.to_dict()
        data['name'] = f'{strategy_class}_kf_{i}'
        data['short_code'] = f'{strategy_class}_kf_{i}'

        if dump_json:
            with open(f'{user_home_path}/backtest_system/backtest_engine/strategies_parameters/{strategy_class}/{strategy_class}_kf_{file_ref}_{i}.json', 'w') as f:
                json.dump(data, f, indent=4, sort_keys=False)
    
    all_datetime_details = pd.concat(datetime_list)
    all_train_stats = pd.concat(train_list)
    all_test_stats = pd.concat(test_list)
    all_valid_stats = pd.concat(valid_list)
    all_params_list = pd.concat(params_list)
    all_fold_list = pd.concat(fold_record_list)

    # params & stats
    frames = [all_params_list, all_train_stats, all_test_stats, all_valid_stats]
    result = pd.concat(frames, axis=1, join="outer")
    print("|------------------------------------|")
    print(f"K Fold Searching Experiment: {file_ref}")
    print(result)
    print("|------------------------------------|")

    # for hiplot result 
    hiploy_analysis_frames = pd.concat([result.reset_index(drop=True), all_fold_list.reset_index(drop=True)], axis=1, join='outer')
    hiploy_analysis_frames['file_ref'] = file_ref
    print("|------------------------------------|")
    print(f"K Fold Searching Experiment: {file_ref}")
    print(hiploy_analysis_frames)
    print("|------------------------------------|")

    # params & datetime info
    frames_datetime = pd.concat([all_params_list.reset_index(drop=True), all_datetime_details.reset_index(drop=True)], axis=1, join='outer')
    print(frames_datetime)
    print("|------------------------------------|")
    print(f"strategy_class: {strategy_class}")
    print("|------------------------------------|")
    print(f"symbol: {symbol}")
    print(f"Reference num start with: {file_ref}")
    print("|------------------------------------|")

    if fold_sep:
        os.makedirs(f'{os.path.dirname(__file__)}/kfold_result/stats_data/{strategy_class}_sep', exist_ok=True)
        os.makedirs(f'{os.path.dirname(__file__)}/kfold_result/params_data/{strategy_class}_sep', exist_ok=True)
        os.makedirs(f'{os.path.dirname(__file__)}/kfold_result/hiplot_analysis_data/{strategy_class}_sep', exist_ok=True)

        result.to_csv(f'{os.path.dirname(__file__)}/kfold_result/stats_data/{strategy_class}_sep/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_stats_df_{file_ref}.csv')
        frames_datetime.to_csv(f'{os.path.dirname(__file__)}/kfold_result/params_data/{strategy_class}_sep/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_params_df_{file_ref}.csv')
        hiploy_analysis_frames.to_csv(f'{os.path.dirname(__file__)}/kfold_result/hiplot_analysis_data/{strategy_class}_sep/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_stats_df_{file_ref}.csv')

        if save_drive:

            # icloud path & onedrive
            icloud_path = f'{user_home_path}/Library/Mobile Documents/com~apple~CloudDocs/kfold_result'
            onedrive = f'{user_home_path}/Library/CloudStorage/OneDrive-Personal'

            os.makedirs(icloud_path + f'/stats_data/seperate/{strategy_class}_sep', exist_ok=True)
            os.makedirs(icloud_path + f'/params_data/seperate/{strategy_class}_sep', exist_ok=True)
            os.makedirs(icloud_path + f'/hiplot_analysis_data/seperate/{strategy_class}_sep', exist_ok=True)

            os.makedirs(onedrive + f'/stats_data/seperate/{strategy_class}_sep', exist_ok=True)
            os.makedirs(onedrive + f'/params_data/seperate/{strategy_class}_sep', exist_ok=True)
            os.makedirs(onedrive + f'/hiplot_analysis_data/seperate/{strategy_class}_sep', exist_ok=True)

            result.to_csv(icloud_path + f'/stats_data/{strategy_class}_sep/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_stats_df_{file_ref}.csv')
            frames_datetime.to_csv(icloud_path + f'/params_data/{strategy_class}_sep/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_params_df_{file_ref}.csv')
            hiploy_analysis_frames.to_csv(icloud_path + f'/hiplot_analysis_data/{strategy_class}_sep/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_stats_df_{file_ref}.csv')
            print('icloud saved')

            result.to_csv(onedrive + f'/stats_data/{strategy_class}_sep/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_stats_df_{file_ref}.csv')
            frames_datetime.to_csv(onedrive + f'/params_data/{strategy_class}_sep/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_params_df_{file_ref}.csv')
            hiploy_analysis_frames.to_csv(onedrive + f'/hiplot_analysis_data/{strategy_class}_sep/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_stats_df_{file_ref}.csv')
            print('onedrive saved')
    
    else:

        os.makedirs(f'{os.path.dirname(__file__)}/kfold_result/stats_data/{strategy_class}', exist_ok=True)
        os.makedirs(f'{os.path.dirname(__file__)}/kfold_result/params_data/{strategy_class}', exist_ok=True)
        os.makedirs(f'{os.path.dirname(__file__)}/kfold_result/hiplot_analysis_data/{strategy_class}', exist_ok=True)

        result.to_csv(f'{os.path.dirname(__file__)}/kfold_result/stats_data/{strategy_class}/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_stats_df_{file_ref}.csv')
        frames_datetime.to_csv(f'{os.path.dirname(__file__)}/kfold_result/params_data/{strategy_class}/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_params_df_{file_ref}.csv')
        hiploy_analysis_frames.to_csv(f'{os.path.dirname(__file__)}/kfold_result/hiplot_analysis_data/{strategy_class}/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_stats_df_{file_ref}.csv')

        if save_drive:

            # icloud path & onedrive
            icloud_path = f'{user_home_path}/Library/Mobile Documents/com~apple~CloudDocs/kfold_result'
            onedrive = f'{user_home_path}/Library/CloudStorage/OneDrive-Personal'

            os.makedirs(icloud_path + f'/stats_data/{strategy_class}', exist_ok=True)
            os.makedirs(icloud_path + f'/params_data/{strategy_class}', exist_ok=True)
            os.makedirs(icloud_path + f'/hiplot_analysis_data/{strategy_class}', exist_ok=True)

            os.makedirs(onedrive + f'/stats_data/{strategy_class}', exist_ok=True)
            os.makedirs(onedrive + f'/params_data/{strategy_class}', exist_ok=True)
            os.makedirs(onedrive + f'/hiplot_analysis_data/{strategy_class}', exist_ok=True)

            result.to_csv(icloud_path + f'/stats_data/{strategy_class}/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_stats_df_{file_ref}.csv')
            frames_datetime.to_csv(icloud_path + f'/params_data/{strategy_class}/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_params_df_{file_ref}.csv')
            hiploy_analysis_frames.to_csv(icloud_path + f'/hiplot_analysis_data/{strategy_class}/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_stats_df_{file_ref}.csv')
            print('icloud saved')

            result.to_csv(onedrive + f'/stats_data/{strategy_class}/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_stats_df_{file_ref}.csv')
            frames_datetime.to_csv(onedrive + f'/params_data/{strategy_class}/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_params_df_{file_ref}.csv')
            hiploy_analysis_frames.to_csv(onedrive + f'/hiplot_analysis_data/{strategy_class}/{strategy_class}_{symbol}_{K_FOLDS}kf_{VALID_PCT}valid_stats_df_{file_ref}.csv')
            print('onedrive saved')

if __name__ == '__main__':
    main()
