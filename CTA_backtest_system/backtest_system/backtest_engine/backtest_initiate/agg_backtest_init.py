import argparse
import configparser
import glob
import itertools
import os
import sys
import warnings
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [18, 16]
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from tabulate import tabulate
import json

sys.path.insert(1, os.path.dirname(__file__) + '/../..')

from backtest_engine.strategy_analyzer import StrategyAnalyzer
from backtest_engine.strategy_list import StrategyList
from backtest_engine.backtest_pool.cerebro_pool import *
from backtest_engine.backtest_pool.kfold_pool import *
from backtest_engine.backtest_pool.candidate_pool import *
from backtest_engine.cerebro_handle.plotter_handler import *

from dealer.engine_data_handler import strategy_action
from dealer.px_handler import *

config = configparser.ConfigParser()
config.read(f'{user_home_path}/backtest_system/config/backtest_config.ini')
spot_dir = config['paths']['spot_data']
uperp_dir = config['paths']['uperp_data']

"""
--pool DEV --time_code all

--pool TEST --time_code all

--pool PROD --time_code all --run_pf True

--pool CRYPTO --time_code select --start 2022-04-01 --end 2022-05-23

"""

def main():

    parser = argparse.ArgumentParser(description='aggregate backtest')
    parser.add_argument('--pool', help="Pool Type")
    parser.add_argument('--start', help="BT start date", required=False)
    parser.add_argument('--end', help="BT end date", required=False)
    parser.add_argument('--time_code', help="time code", required=True)
    parser.add_argument('--need_sign', help="need_sign", default=False)
    # for portfolio
    parser.add_argument('--run_pf', help="run portfolio bt", default=False)
    parser.add_argument('--run_cor', help="run correlation metric", default=False)

    args = parser.parse_args()
    start = args.start
    end = args.end
    pool = args.pool
    run_pf = args.run_pf
    time_code = args.time_code
    run_cor = args.run_cor
    need_sign = args.need_sign

    if pool == DEV:
        strategy_list = CRYPTO_DEV_POOL
    elif pool == TEST:
        strategy_list = TEST_POOL
    elif pool == PROD:
        strategy_list = PROD_CRYPTO_POOL
    elif args.pool == KFOLD:
        strategy_list = KFOLD_POOL
    elif args.pool == CANDIDATE:
        strategy_list = CANDIDATE_POOL
    else:
        raise('not yet ready or sth wrong...')
    
    all_summary = []
    key_summary = []
    warmup_summary = []
    all_return = pd.DataFrame()
    code_name_all_return = pd.DataFrame()
    trade_signal_dict = {}
    plot_list = []

    key_func = lambda x: x[2] + '|' + x[3] + '|' + x[4] + '|' + x[5]
    for key, group in itertools.groupby(strategy_list, key_func):
        exchange, prod_type, symbol, asset_class = key.split('|')
        strategy_list = list(group)
        today_date = datetime.today().strftime('%Y-%m-%d')
       
        strat_obj_list = []
        for strategy_tuple in strategy_list:
            strategy_class = strategy_tuple[0]
            strategy_config = strategy_tuple[1]
            asset_class = strategy_tuple[5]

            if args.pool == CANDIDATE:
                # read config
                with open(f'{user_home_path}/backtest_system/backtest_engine/strategies_parameters/{strategy_class}/candidate_select/{strategy_config}.json') as f:
                    config = json.loads(f.read())
            
            else:
                # read config
                with open(f'{user_home_path}/backtest_system/backtest_engine/strategies_parameters/{strategy_class}/{strategy_config}.json') as f:
                    config = json.loads(f.read())
            
            if need_sign:
                designer = config['strategy_details']['designer']
                design_date = config['strategy_details']['design_date']
                status = config['strategy_details']['status'] 
            else: 
                pass           

            data_center_dir = f'{user_home_path}/backtest_system/data_center/cerebro_engine'

            os.makedirs(data_center_dir, exist_ok=True)
            os.makedirs(data_center_dir + "/agg_bt_result", exist_ok=True)

            data_df = uperp_data_import(symbol, uperp_dir)
            selected_data_df = bar_slicer(time_code, data_df, today_date, start, end)

            strat_obj_list.append(strategy_config)
            print("|-------------------------------------------|")
            print(f"{strategy_config}                           ")
            print("|-------------------------------------------|")

            # get the strategy
            if pool == KFOLD:
                strategy, strat_config = StrategyList.get_strategy(strategy_class, f"{strategy_config}.json", selected_data_df, return_config=True, kfold=True)
            elif pool == CANDIDATE:
                strategy, strat_config = StrategyList.get_strategy(strategy_class, f"{strategy_config}.json", selected_data_df, return_config=True, candidate=True)
            else:
                strategy, strat_config = StrategyList.get_strategy(strategy_class, f"{strategy_config}.json", selected_data_df, return_config=True)

            cerebro = StrategyAnalyzer(strategy, selected_data_df, output_path=data_center_dir, output_file_suffix=None)
            cerebro.backtest_analysis_report(save_result=False, print_result=True, show_plots=False, y_log_scale=False)       

            # metric_df
            metric_df = cerebro.data_df
            timeframe_metric = metric_df.index.to_series().diff()
            metric_df['trading_sys_time'] = metric_df.index + timeframe_metric[-1] # add the trading_sys_time can compare to the backtest time
            os.makedirs(data_center_dir + f"/strategy_data/{strategy_class}/metric_df/", exist_ok=True)
            metric_df.to_csv(data_center_dir + f"/strategy_data/{strategy_class}/metric_df/{strategy_config}_metric_df.csv")
            print('metric_df saved!')   
            print('----------------')  

            # also add calculate the warm-up period for deployment system
            warmup_bar_unit = metric_df.isnull().sum().max()
            warmup_length = metric_df.index[warmup_bar_unit] - metric_df.index[0]

            # strategy_df
            strategy_df = cerebro.strategy_df
            timeframe_strattegy_df = strategy_df.index.to_series().diff()
            strategy_df['trading_sys_time'] = strategy_df.index + timeframe_strattegy_df[-1]
            strategy_df.loc[:, 'action'] = strategy_df.apply(strategy_action, axis = 1)
            os.makedirs(data_center_dir + f"/strategy_data/{strategy_class}/strategy_df/", exist_ok=True)
            strategy_df.to_csv(data_center_dir + f"/strategy_data/{strategy_class}/strategy_df/{strategy_config}_strategy_df.csv")
            print('strategy_df saved!')
            print('----------------')  

            # get trade pnl ingrident
            strategy = cerebro.strategy
            strategy_df = cerebro.strategy_df
            trade_signals = strategy.trade_signals
            trade_details = strategy.trade_details

            strategy_summary, key_strategy_summary, _, _, _ = cerebro.stats_calculate(
                strategy_df=strategy_df, 
                trade_signals=trade_signals, 
                from_dt=None,
                return_field="return",
                method="cumsum",
                filter_no_trade_day=False,
                trade_details=trade_details)

            trade_pnl_df = strategy_summary['trade_pnl_df']
            os.makedirs(data_center_dir + f'/strategy_data/{strategy_class}/trade_pnl_df/', exist_ok=True)
            trade_pnl_df.to_csv(data_center_dir + f'/strategy_data/{strategy_class}/trade_pnl_df/{strategy_config}_trade_pnl_df.csv')
            print('trade_pnl_df saved!')
            print('----------------')  

            if need_sign:
                print(f"The strategy designer is {designer}")
                print(f'Designed in {design_date}')
                print('----------------')  
                print(f"Strategy Type: {status}")
                print('----------------')  
            else:
                pass

            day_return, strategy_summary, _, _, benchmark_metrics = cerebro.backtest_analysis_report(print_result=False, save_result=False, show_plots=False)
            # out put html plot
            return_mdd, _ = cerebro.plotter.plot_return_series_interactive(
                day_return=day_return,
                bm_cum_return=benchmark_metrics['bm_cum_return'],
                show=False,
                y_log_scale=False
            )
            monthly_return_fig = cerebro.plotter.plot_monthly_return_interactive(day_return=day_return, show=False)
            weekly_return_fig = cerebro.plotter.plot_weekly_return_interactive(day_return=day_return, show=False)
            plot_list.append([strategy_config, return_mdd, monthly_return_fig, weekly_return_fig])
            seperate_plot_list = [strategy_config, return_mdd, monthly_return_fig, weekly_return_fig] # for seperate strategy

            # generate seperate_strategy_hist_html
            os.makedirs(data_center_dir +  "/seperate_strategy_hist_html" + f"/{strategy_class}", exist_ok=True) 
            with open(data_center_dir +  "/seperate_strategy_hist_html" + f"/{strategy_class}/{strategy_config}.html",'w') as dashboard:
                dashboard.write("<html><head></head><body>" + "\n")
                html_str = f"""
                            <div style="font-family: monospace, monospace;">
                                <p>{seperate_plot_list[0]}</p>
                            </div>
                        """
                dashboard.write(html_str)
                inner_html = seperate_plot_list[1].to_html().split('<body>')[1].split('</body>')[0]
                monthly_returns_html = seperate_plot_list[2].to_html().split('<body>')[1].split('</body>')[0]
                weekly_returns_html = seperate_plot_list[3].to_html().split('<body>')[1].split('</body>')[0]
                dashboard.write(inner_html)
                dashboard.write(monthly_returns_html)
                dashboard.write(weekly_returns_html)
                dashboard.write("</body></html>" + "\n")

            day_return.rename(columns={'return': strat_config['short_code']}, inplace=True)
            summary_dict = {"strategy_name": strategy_config, "date_start": selected_data_df.index[0], "date_end": selected_data_df.index[-1]}
            key_summary_dict = {"strategy_name": strategy_config, "date_start": selected_data_df.index[0], "date_end": selected_data_df.index[-1]}
            warmup_dict = {"strategy_name": strategy_config, "backtesting name": strategy_config, "warmup_bar_unit": warmup_bar_unit, "warmup_time": warmup_length}
            summary_dict.update(strategy_summary)
            key_summary_dict.update(key_strategy_summary)

            all_return = pd.merge(all_return, day_return[[strat_config['short_code']]], left_index=True, right_index=True, how='outer')
            code_name_all_return = pd.merge(code_name_all_return, day_return[[strat_config['short_code']]], left_index=True, right_index=True, how='outer')
            # code_name_all_return
            code_name_all_return = code_name_all_return.rename(columns={f'{strategy_config}': f'{strategy_config}'})
            all_summary.append(summary_dict)
            warmup_summary.append(warmup_dict)
            key_summary.append(key_summary_dict)

            trade_signal = cerebro.strategy_df['trade_signal'].resample('1min').last().fillna(0)
            trade_signal_dict[strategy_config] = trade_signal

    start_bt_time = selected_data_df.index[0].strftime("%Y_%m_%d")
    latest_bt_time = selected_data_df.index[-1].strftime("%Y_%m_%d")
    dt_suffix = latest_bt_time
    print(f"Strategy html updated from {start_bt_time} to {latest_bt_time}!!")
    all_trading_signals = pd.DataFrame(trade_signal_dict).fillna(0)
    all_trading_signals.index.name = 'datetime'
    all_return.index.name = 'date'
    code_name_all_return.index.name = 'date'
    # naming for backtest
    bt_result = pd.DataFrame(all_summary)
    bt_result = bt_result.transpose()
    warmup_df = pd.DataFrame(warmup_summary)

    os.makedirs(data_center_dir +  f"/warmup_period", exist_ok=True)
    warmup_df.to_csv(data_center_dir +  f"/warmup_period/warmup_period.csv")
    key_bt_result = pd.DataFrame(key_summary)
    key_bt_result = key_bt_result.transpose()
    # round up 
    for index in key_bt_result.index:
        for i in  key_bt_result.columns:
            if isinstance((key_bt_result.loc[index])[i], float) == True:
                (key_bt_result.loc[index])[i] = round((key_bt_result.loc[index])[i],3)
    print(tabulate(key_bt_result, headers='keys', tablefmt='github'))
    print('----------')

    # print(bt_result)
    latest_bt_time = data_df.index[-1]
    dt_suffix = latest_bt_time.strftime("%Y-%m-%d")
    print(f"Portfolio updated to {dt_suffix}!!")

    os.makedirs(data_center_dir + '/stats', exist_ok=True)
    os.makedirs(data_center_dir + '/agg_bt_result', exist_ok=True)
    os.makedirs(data_center_dir + '/equal_weight_portfolio', exist_ok=True)

    bt_result.to_csv(data_center_dir + '/stats/stats.csv')
    key_bt_result.to_csv(data_center_dir + '/stats/key_stats.csv')
    key_bt_result.transpose().to_excel(data_center_dir + '/stats/key_stats.xlsx')
    all_return.to_csv(data_center_dir + '/agg_bt_result/return_series.csv')
    code_name_all_return.to_csv(data_center_dir + '/agg_bt_result/code_name_return_series.csv')
    all_trading_signals.to_csv(data_center_dir + '/agg_bt_result/trading_signals.csv')

    if run_cor:
        print("-----------------")
        print("Running Correlation Metric")
        print("-----------------")
        cor_return = pd.read_csv(data_center_dir + '/agg_bt_result/return_series.csv', index_col = 'date')
        cor_return.index = pd.to_datetime(cor_return.index)
        cor_return = cor_return.dropna()
        print("correlation return start date")
        print(min(cor_return.index))
        print("-----------------")
        print("correlation end date")
        print(max(cor_return.index))
        print("-----------------")
        for col in cor_return.columns:
            mean = cor_return[col].mean()
            var = cor_return[col].var()
            print(f'{col}: {mean=:.6f}, {var=:.6f}')
        corr = cor_return.corr()
        cor_metric = sns.heatmap(corr)
        cor_metric.xaxis.tick_top() # x axis on top
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        cor_metric_fig = cor_metric.get_figure()
        os.makedirs(data_center_dir + '/correlation_analysis', exist_ok=True)
        cor_metric_fig.savefig(data_center_dir + "/correlation_analysis/correlation_metric.png") 
        print("cor_metric_fig saved")
    if run_pf: 
        print('----------')
        print('Portfolio contain...')
        for strat in all_return.columns:
                print(strat)
        print('----------')
        pf_return_df = all_return
        pf_return_df.index = pd.to_datetime(pf_return_df.index)
        pf_return_df = pf_return_df.dropna() # start from the join warmup period period

        start = pf_return_df.index[0].strftime("%Y-%m-%d")
        end = pf_return_df.index[-1].strftime("%Y-%m-%d")
        main_title = 'All Portfolio Strategy Return -'
        pf_title = main_title + f' {start} to {end}'
        pf_fig = go.Figure()
        for strat in pf_return_df.columns:
            pf_fig.add_trace(go.Scatter(x=pf_return_df.index, y=pf_return_df[strat].cumsum(), name=strat))
            pf_fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01), width=1000, height=800, title=f"{pf_title}") 

        pf_fig.write_image(data_center_dir + "/equal_weight_portfolio/all_strategies_plot.png") 
        portfolio_all_return, portfolio_stats = StrategyAnalyzer.cal_ew_portfolio_stats_from_daily_nav(all_return) # portfolio equal-weight backtest
        os.makedirs(data_center_dir + "/equal_weight_portfolio", exist_ok=True)
        # save the file  
        portfolio_all_return.to_csv(data_center_dir + "/equal_weight_portfolio/ew_pf_all_return.csv")
        portfolio_stats.to_csv(data_center_dir + "/equal_weight_portfolio/ew_portfolio_stats_table.csv")

        ew_main_title = 'Equal Weight Portfolio -'
        ew_pf_title = ew_main_title + f' {start} to {end}'
        ew_fig = go.Figure()
        ew_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, specs=[[{"type": "table"}],[{"type": "scatter"}]])
        ew_fig.add_trace(go.Scatter(x=portfolio_all_return.index, y=(portfolio_all_return.pf_nav_from_1)-1))
        ew_fig.add_trace(
            go.Table(
                header=dict(
                    values=["Portfolio_Stats", "value"],
                    font=dict(size=10),
                    align="right"
                ),
                cells=dict(
                    values=[portfolio_stats.index, portfolio_stats.value],
                    align = "right")
            ),
            row=1, col=1
        )
        ew_fig.update_layout(width=1000, height=1500,title=ew_pf_title)
        # save the plot
        os.makedirs(data_center_dir + "/equal_weight_portfolio/ew_plot", exist_ok=True)
        ew_fig.write_image(data_center_dir + "/equal_weight_portfolio/ew_plot/ew_plot.png") 
        ew_fig.show()
    else: 
        print("Skip the portfolio backtesting...")
    # for agg html report
    os.makedirs(data_center_dir +  "/agg_bt_hist_html", exist_ok=True) # create folder not exist
    with open(data_center_dir + "/agg_bt_hist_html" + "/agg_bt.html",'w') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig_info in plot_list:
            html_str = f"""
                        <div style="font-family: monospace, monospace;">
                            <p>{fig_info[0]}</p>
                        </div>
                    """
            dashboard.write(html_str)
            inner_html = fig_info[1].to_html().split('<body>')[1].split('</body>')[0]
            monthly_returns_html = fig_info[2].to_html().split('<body>')[1].split('</body>')[0]
            weekly_returns_html = fig_info[3].to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
            dashboard.write(monthly_returns_html)
            dashboard.write(weekly_returns_html)
        dashboard.write("</body></html>" + "\n")
        print("Html saved in: " + data_center_dir + "/agg_bt_hist_html" + "/agg_bt.html")
        webbrowser.open("file://" + data_center_dir + "/agg_bt_hist_html" + "/agg_bt.html") # show after running
    
    # plot the return NAV in the pool
    return_df = pd.read_csv(data_center_dir + '/agg_bt_result/return_series.csv', index_col = 'date')
    return_df.index = pd.to_datetime(return_df.index)
    return_df = return_df.dropna()
    start = return_df.index[0].strftime("%Y-%m-%d")
    end = return_df.index[-1].strftime("%Y-%m-%d")

    if args.pool == KFOLD:
        print("Skip the original aggbt since doing the kfold...")
    
    else:

        Title = 'NAV Performance -' + f' {start} to {end}'
        agg_return_fig = go.Figure()

        for strat in return_df.columns:

            if len(return_df.columns) > 5:
                agg_return_fig.add_trace(go.Scatter(x=return_df.index, y=return_df[strat].cumsum(), name=strat))
                agg_return_fig.update_layout(legend=dict(yanchor="top",y=0.99, xanchor="left", x=0.01), width=1000, height=800, title=f"{Title}") 
            else:
                agg_return_fig.add_trace(go.Scatter(x=return_df.index, y=return_df[strat].cumsum(), name=strat))
                agg_return_fig.update_layout(legend=dict(yanchor="top",y=0.99, xanchor="left", x=0.01), width=1000, height=800, title=f"{Title}") 
                agg_return_fig.add_vrect(x0="2021-12-01", x1=f"{today_date}", annotation_text="out-sample", annotation_position="top left", fillcolor="green", opacity=0.05, line_width=0)
                agg_return_fig.add_vrect(x0=f"{start}", x1="2021-12-01", annotation_text="in-sample", annotation_position="top right", fillcolor="blue", opacity=0.05, line_width=0)
        # agg_return_fig.show()
        os.makedirs(data_center_dir +  "/agg_return_fig", exist_ok=True)
        agg_return_fig.write_image(data_center_dir + "/agg_return_fig/agg_return_fig.png") 
        agg_return_fig.show()

    # draw the kfold validate picture in different period
    if args.pool == KFOLD:

        valid_pct = 0.1
        folds = 5
        kfold_symbol = input("Please enter the kfold_symbol: ")
        kfold_ref_num = input("Please enter the kfold_ref_num: ")

        date_label_path = f'{user_home_path}/backtest_system/backtest_engine/kfold_search/kfold_result/params_data/{strategy_class}/{strategy_class}_{kfold_symbol}_{folds}kf_{valid_pct}valid_params_df_{kfold_ref_num}.csv'
        date_label_df = pd.read_csv(date_label_path)
        print("The path of the kfold result is :" + date_label_path)

        valid_start = datetime.strptime(date_label_df['valid_data_start'][0], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        valid_end = datetime.strptime(date_label_df['valid_data_end'][0], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        # fold-1
        kfold_1_start = datetime.strptime(date_label_df['train_start'][0], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_1_end = datetime.strptime(date_label_df['train_end'][0], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_1_test_start = datetime.strptime(date_label_df['test_start'][0], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_1_test_end = datetime.strptime(date_label_df['test_end'][0], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        # fold-2
        kfold_2_start = datetime.strptime(date_label_df['train_start'][1], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_2_end = datetime.strptime(date_label_df['train_end'][1], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_2_test_start = datetime.strptime(date_label_df['test_start'][1], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_2_test_end = datetime.strptime(date_label_df['test_end'][1], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        # fold-3
        kfold_3_start = datetime.strptime(date_label_df['train_start'][2], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_3_end = datetime.strptime(date_label_df['train_end'][2], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_3_test_start = datetime.strptime(date_label_df['test_start'][2], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_3_test_end = datetime.strptime(date_label_df['test_end'][2], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        # fold-4
        kfold_4_start = datetime.strptime(date_label_df['train_start'][3], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_4_end = datetime.strptime(date_label_df['train_end'][3], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_4_test_start = datetime.strptime(date_label_df['test_start'][3], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_4_test_end = datetime.strptime(date_label_df['test_end'][3], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        # fold-5
        kfold_5_start = datetime.strptime(date_label_df['train_start'][4], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_5_end = datetime.strptime(date_label_df['train_end'][4], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_5_test_start = datetime.strptime(date_label_df['test_start'][4], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        kfold_5_test_end =  datetime.strptime(date_label_df['test_end'][4], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')

        agg_kfold_fig_title = f'KFOLD Validation - {strategy_class} {symbol}'
        agg_kfold_fig = go.Figure()
        for strat in return_df.columns:
            agg_kfold_fig.add_trace(go.Scatter(x=return_df.index, y=return_df[strat].cumsum(), name=strat))
            agg_kfold_fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01), width=1000, height=800, title=f"{agg_kfold_fig_title}")
            # valid 
            agg_kfold_fig.add_vrect(x0=valid_start, x1=f"{today_date}", annotation_text="Validate", annotation_position="top left", fillcolor="green", opacity=0.04, line_width=0)
            agg_kfold_fig.add_vline(x=valid_start, line_width=1, line_dash="dash", line_color="black")
            # fold-1
            agg_kfold_fig.add_vline(x=kfold_1_end, line_width=1, line_dash="dash", line_color="black")
            # fold-2
            agg_kfold_fig.add_vline(x=kfold_2_end, line_width=1, line_dash="dash", line_color="black")
            # fold-3
            agg_kfold_fig.add_vline(x=kfold_3_end, line_width=1, line_dash="dash", line_color="black")
            # fold-4
            agg_kfold_fig.add_vline(x=kfold_4_end, line_width=1, line_dash="dash", line_color="black")
            # fold-5
            agg_kfold_fig.add_vline(x=kfold_5_end, line_width=1, line_dash="dash", line_color="black")

        # agg_kfold_fig.show()
        os.makedirs(data_center_dir + f"/kfold/kfold_return_plot/{strategy_class}", exist_ok=True)
        agg_kfold_fig.write_image(data_center_dir + f"/kfold/kfold_return_plot/{strategy_class}/{strategy_class}_{symbol}_{kfold_ref_num}_kfold_validate.png") 
        
        os.makedirs(data_center_dir +  "/agg_strategy_return_fig_html", exist_ok=True)
        with open(data_center_dir +  "/agg_strategy_return_fig_html" + "/kfold_validate_usage.html",'w') as dashboard:
            dashboard.write("<html><head></head><body>" + "\n")
            html_str = f"""
                        <div style="font-family: monospace, monospace;">
                            <p>Kfold_Validate</p>
                        </div>
                    """
            dashboard.write(html_str)
            inner_html = agg_kfold_fig.to_html()
            dashboard.write(inner_html)
            dashboard.write("</body></html>" + "\n")
            webbrowser.open("file://" + data_center_dir +  "/agg_strategy_return_fig_html/kfold_validate_usage.html")


if __name__ == '__main__':
    main()










