import os 
from os.path import isfile, join
import sys
from pathlib import Path
import argparse
from weakref import ref
import webbrowser
from datetime import datetime

# locate the path 
sys.path.insert(1, os.path.dirname(__file__) + '/../..')
user_home_path = str(Path.home())

import pandas as pd
import plotly.graph_objs as go
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

"""
--strategy_logic SmrcExpStrategy --ref_num 7777
"""

def main():

    parser = argparse.ArgumentParser(description='tuning return plot')
    parser.add_argument('--ref_num', help="is_icloud", required=True)
    parser.add_argument('--strategy_logic', help="strategy_logic", required=True)
    # parser.add_argument('--symbol', help="symbol", required=True)

    plateau_search_file = "/Users/chouwilliam/Library/Mobile Documents/com~apple~CloudDocs/plateau_search"
    local_plateau_search_file = '/Users/chouwilliam/backtest_system/backtest_engine/strategy_tuner/plateau_search'

    os.makedirs(plateau_search_file, exist_ok=True)
    os.makedirs(local_plateau_search_file, exist_ok=True)

    args = parser.parse_args()
    ref_num = int(args.ref_num)
    strategy_logic = args.strategy_logic

    slicing_df_path = '/Users/chouwilliam/backtest_system/backtest_engine/strategy_tuner/plateau_search/slicing_hiplot/hiplot_selected_data.csv' # fixed file name 
    
    # read the selected params and get the uid in slicing_df_path
    slicing_df = pd.read_csv(slicing_df_path) # fixed and replace
    slicing_df_uid = list((slicing_df['uid']))

    print('--------select----------')
    print(slicing_df_uid)
    print('--------select----------')

    plateau_search_file = "/Users/chouwilliam/Library/Mobile Documents/com~apple~CloudDocs/plateau_search"

    experiment_folder = plateau_search_file + '/iterate_return' + f'/{ref_num}'
    os.makedirs(experiment_folder, exist_ok=True)

    # get the all tuning return data file 
    list_all_candidate = os.listdir(experiment_folder)
        
    # put all the uit make a list with the same file name with the real NAV
    target_candidate = []
    for i in slicing_df_uid:
        i = str(i) + '_day_return.csv'
        target_candidate.append(i)

    all_return = pd.DataFrame()
    list_all_candidate = set(list_all_candidate)
    for i in tqdm(target_candidate):
        if i in list_all_candidate:
            candidate_df = pd.read_csv(experiment_folder + f'/{i}') 
            candidate_df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
            candidate_df = candidate_df[['datetime', 'return']]
            candidate_df.rename(columns={'return': f'{i}'.replace('_day_return.csv', '')}, inplace=True)
            candidate_df.set_index('datetime', inplace=True)
            all_return = pd.merge(all_return, candidate_df, left_index=True, right_index=True, how='outer')

    # get the all_return file and save it into new file in icloud & local
    aggregate_iterate_result = plateau_search_file + f'/aggregate_iterate_return/{strategy_logic}_{ref_num}'
    
    # saved in icloud
    os.makedirs(aggregate_iterate_result, exist_ok=True)
    all_return.to_csv(aggregate_iterate_result + f'/agg_iterate_return_{strategy_logic}_{ref_num}.csv')

    # saved in local
    local_save_path = local_plateau_search_file + '/aggregate_result'
    os.makedirs(local_save_path, exist_ok=True)
    all_return.to_csv(local_save_path + f'/agg_iterate_return_{strategy_logic}_{ref_num}.csv')

    # drawing the NAV
    all_return = all_return.dropna()
    start = all_return.index[0]
    end = all_return.index[-1]
    Title = 'Plateau Searching -' + f'{strategy_logic}_{ref_num} {start} to {end}'
    plateau_search_fig = go.Figure()
    for strat in tqdm(all_return.columns):
        plateau_search_fig.add_trace(go.Scatter(x=all_return.index, y=all_return[strat].cumsum(), name=strat))
        plateau_search_fig.update_layout(legend=dict(yanchor="top",y=0.99, xanchor="left", x=0.01), width=1000, height=800, title=f"{Title}") 
    
    # plot saving local
    local_plot_save_path = local_plateau_search_file + f'/plateau_plot/{strategy_logic}'
    os.makedirs(local_plot_save_path + f"/{strategy_logic}_{ref_num}", exist_ok=True)
    plateau_search_fig.write_image(local_plot_save_path + f"/{strategy_logic}_{ref_num}/{strategy_logic}_{ref_num}.png")
    print(f'Plateau plot saved at local: {local_plot_save_path} + "/{strategy_logic}_{ref_num}/{strategy_logic}_{ref_num}.png"')

    plateau_search_fig.show()
    

if __name__ == '__main__':
    main()