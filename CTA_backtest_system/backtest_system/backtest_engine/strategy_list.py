import json
import os
import sys
from pathlib import Path
from copy import copy

sys.path.insert(1, os.path.dirname(__file__) + '/../..')
user_home_path = str(Path.home())

from backtest_engine.strategies import *


from backtest_engine.strategy_analyzer import StrategyAnalyzer

class StrategyList:

    def __init__(self):
        pass

    @staticmethod
    def get_strategy(strategy_logic, strategy_config, data_df, start_date, end_date, return_config=False, prod=False, plateau=False, kfold=False , candidate=False):

        try:
            if prod:
                with open(f'./backtest_engine/PROD_Strategies_Parameters/{strategy_logic}/{strategy_config}') as f:
                    config = json.loads(f.read())
                strategy = eval(strategy_logic)(config, data_df,)
            elif plateau:
                with open(f'./backtest_engine/strategies_parameters/{strategy_logic}/plateau_search/{strategy_config}') as f:
                    config = json.loads(f.read())
                strategy = eval(strategy_logic)(config, data_df)
            elif candidate:
                with open(f'./backtest_engine/strategies_parameters/{strategy_logic}/candidate_select/{strategy_config}') as f:
                    config = json.loads(f.read())
                strategy = eval(strategy_logic)(config, data_df)
            elif kfold:
                with open(f'./backtest_engine/strategies_parameters/{strategy_logic}/kfold_search/{strategy_config}') as f:
                    config = json.loads(f.read())
                strategy = eval(strategy_logic)(config, data_df)
            else:
                with open(f'./backtest_engine/strategies_parameters/{strategy_logic}/{strategy_config}') as f:
                    config = json.loads(f.read())
                strategy = eval(strategy_logic)(config, data_df, start_date, end_date)
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Config file directory not found: {e}')
        except NameError as e:
            raise NotImplementedError(f'Strategy class not implemented, or not found in strategy directory: {e}')
        if return_config:
            return strategy, config
        else:
            return strategy
