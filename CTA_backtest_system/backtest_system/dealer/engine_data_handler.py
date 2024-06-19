import pandas as pd 
import numpy as np
import sys
from pathlib import Path
import os

sys.path.insert(1, os.path.dirname(__file__) + '/../..')
user_home_path = str(Path.home())

from backtest_engine.strategy_analyzer import StrategyAnalyzer
from backtest_engine.strategy_list import StrategyList

def cerebro_init(strategy_logic, strategy_config, data_df, start_date, end_date):

    # cerebro init
    strategy_selected = StrategyList.get_strategy(
        strategy_logic=strategy_logic, 
        strategy_config=f"{strategy_config}.json", 
        data_df=data_df,
        start_date = start_date,
        end_date = end_date
        )

    cerebro = StrategyAnalyzer(strategy_selected)

    cerebro.backtest_analysis_report(
        save_result=True, 
        print_result=True, 
        show_plots=False, 
        y_log_scale=False)

    return cerebro

def strategy_action(row): # for strategy_df add action using apply
    if row.trade_signal == 1 and row.position == 0:
        return 'open_long'
    elif row.trade_signal == -1 and row.position == 0:
        return 'open_short'
    elif row.trade_signal == -1 and row.position > 0:    
        return 'close_long'
    elif row.trade_signal == 1 and row.position < 0:
        return 'close_short'
    else:
        return 0
    
def generate_backtest_result_csv(cerebro, asset_class, strategy_logic, strategy_config, save_dir):
    metric_df_handler(cerebro, asset_class, strategy_logic, strategy_config, save_dir)
    strategy_df_handler(cerebro, asset_class, strategy_logic, strategy_config, save_dir)
    trade_pnl_handler(cerebro, asset_class, strategy_logic, strategy_config, save_dir)
    
    
def metric_df_handler(cerebro, asset_class, strategy_logic, strategy_config, save_dir):
    metric_df = cerebro.data_df
    timeframe_metric = metric_df.index.to_series().diff()
    metric_df['trading_sys_time'] = metric_df.index + timeframe_metric[-1] # add the trading_sys_time can compare to the backtest time
    
    os.makedirs(save_dir + f"/{asset_class}/{strategy_logic}/metric_df/", exist_ok=True)
    metric_df.to_csv(save_dir + f"/{asset_class}/{strategy_logic}/metric_df/{strategy_config}_metric_df.csv")
    print('metric_df saved!')

def strategy_df_handler(cerebro, asset_class, strategy_logic, strategy_config, save_dir):
    strategy_df = cerebro.strategy_df
    timeframe_strategy_df = strategy_df.index.to_series().diff()
    strategy_df['trading_sys_time'] = strategy_df.index + timeframe_strategy_df[-1]
    strategy_df.loc[:, 'action'] = strategy_df.apply(strategy_action, axis = 1)
    
    os.makedirs(save_dir + f"/{asset_class}/{strategy_logic}/strategy_df/", exist_ok=True)
    strategy_df.to_csv(save_dir + f"/{asset_class}/{strategy_logic}/strategy_df/{strategy_config}_strategy_df.csv")
    print('strategy_df saved!')
    

def trade_pnl_handler(cerebro, asset_class, strategy_logic, strategy_config, save_dir):
    strategy = cerebro.strategy
    strategy_df = cerebro.strategy_df
    trade_signals = strategy.trade_signals
    trade_details = strategy.trade_details

    strategy_summary_pnl, _, _, _, _ = cerebro.stats_calculate(
        strategy_df=strategy_df, 
        trade_signals=trade_signals, 
        from_dt=None,
        return_field="return",
        method="cumsum",
        filter_no_trade_day=False,
        trade_details=trade_details)
    trade_pnl_df = strategy_summary_pnl['trade_pnl_df']
    
    os.makedirs(save_dir + f'/{asset_class}/{strategy_logic}/trade_pnl_df/', exist_ok=True)
    trade_pnl_df.to_csv(save_dir + f'/{asset_class}/{strategy_logic}/trade_pnl_df/{strategy_config}_trade_pnl_df.csv')
    print('trade_pnl_df saved!')

def save_strategy_html_report(asset_class, strategy_logic, strategy_config, data_df, save_dir, start_date, end_date):
    # to-do

    cerebro = cerebro_init(strategy_logic, strategy_config, data_df, start_date, end_date)
    # html saving path
    cerebro.backtest_analysis_report(save_result=True, print_result=True, show_plots=False, y_log_scale=False)

def cerebro_summary_output(strategy_logic, strategy_config, data_df, start_date, end_date):

    cerebro = cerebro_init(strategy_logic, strategy_config, data_df, start_date, end_date)
    strategy_summary, key_strategy_summary, trade_dist, rolling_sharpe, day_return = cerebro.backtest_analysis_report(save_result=False, print_result=False, show_plots=False, y_log_scale=False)

    return strategy_summary, key_strategy_summary, trade_dist, rolling_sharpe, day_return




