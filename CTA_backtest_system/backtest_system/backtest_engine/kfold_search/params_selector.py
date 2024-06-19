import os
import sys
from pathlib import Path

import pandas as pd

def param_set_determine(file_path):

    param_set = pd.read_csv(file_path)
    param_set = param_set[param_set['trade_per_day_train'] >= 0.5]
    param_set = param_set[param_set['sharpe_train'] >= 1.5]
    param_set = param_set[param_set['Ann_PnL/MDD_train'] >= 2.0]
    param_set = param_set[param_set['max_drawdown_train'] <= 0.3]
    
    # rank
    df = param_set.nlargest(5, 'sharpe_train')

    if len(df) == 0: # protection in case selection failed 
        print("selection failed, directly get the best sharpe train and continue")
        param_set_second = pd.read_csv(file_path)
        df = param_set_second.nlargest(5, 'sharpe_train')
      
    return df.iloc[0]

# adding this just for discussion! 
def param_set_determine_snoop(file_ref): 

    param_set = pd.read_csv(file_ref)
    param_set = param_set[param_set['trade_per_day_train'] >= 0.4]
    param_set = param_set[param_set['sharpe_train'] >= 1.0]
    param_set = param_set[param_set['ann_return/MDD_train'] >= 2.0]
    param_set = param_set[param_set['max_drawdown_train'] <= 0.4]

    # peak the test 
    param_set['sharpe_train_test_avg'] = (param_set['sharpe_train'] + param_set['sharpe_test'])/2
    param_set = param_set[param_set['sharpe_test'] >= 1.0]
    param_set = param_set[param_set['max_drawdown_test'] <= 0.4]
    param_set = param_set[param_set['sharpe_train_test_avg'] >= 1.5]
    
    # rank
    df = param_set.nlargest(10, 'sharpe_train')

    if len(df) == 0: # protection in case selection failed 
        print("selection failed, directly get the best sharpe train and continue")
        param_set_second = pd.read_csv(file_ref)
        df = param_set_second.nlargest(5, 'sharpe_train')
            
    return df.iloc[0]