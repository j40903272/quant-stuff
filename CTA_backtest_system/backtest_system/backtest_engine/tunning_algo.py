import csv
import os
from pathlib import Path
user_home_path = str(Path.home())
import numpy as np
import optuna

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from numpy import random
from scipy.stats import norm
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from .backtest_core.backtester import BackTester
from backtest_engine.exceptions import ParamError

import configparser
config = configparser.ConfigParser()
config.read(f"./config/tuning_config.ini")

class BaseParamSearcher:

    def __init__(self, strat_analyzer, data=None, train_percent=None, train_data=None, test_data=None, spot_data=None, all_data=None):
        self.strat_analyzer = strat_analyzer
        self._strategy = strat_analyzer.strategy
        self._portfolio_params = strat_analyzer.portfolio_params

        if data is not None and train_percent is not None:
            self.data = data
            self.train_percent = train_percent
            self.train = data[:int(len(data) * train_percent)].copy() if train_percent is not None else data
            self.test = data[int(len(data) * train_percent):].copy() if train_percent is not None else data

        elif train_data is not None and test_data is not None:
            self.data = None
            self.train_percent = None
            self.train = train_data
            self.test = test_data
            self.all = all_data

        if spot_data is not None:
            self.spot = spot_data

        self.funding_rates_series = None
        # keys and params for searches
        try:
            self.prep_data_params = self._strategy.search_dict["prepare_data_param"]
            self.trading_data_params = self._strategy.search_dict["trading_data_param"]
            self.prep_data_params_types = self._strategy.search_dict["prepare_data_param_type"]
            self.trading_data_params_types = self._strategy.search_dict["trading_data_param_type"]
        except:
            raise ParamError('Not all four required param dicts are in strategy config: <prepare/traing>_data_param and their types.')

        self._resample_ps()
        self.headers = (["resample"] + list(self.prep_data_params.keys()) + list(self.trading_data_params.keys())
                        + ["sharpe", "sortino", "annualized_returns", "trade_per_day", "total_trades_number", "max_drawdown", "ann_return/MDD", "win_rate", "win_loss_ratio"]
                        )
        self.prep_keys = list(self.prep_data_params.keys())
        self.trade_keys = list(self.trading_data_params.keys())
        self.output_path = None

    def init_writer(self, path):
        self.fout = open(self.output_path, "w", newline="")
        self.writer = csv.writer(self.fout)

    def close_writer(self):
        self.fout.close()

    def write(self, w):
        self.writer.writerow(w)

    def _resample_ps(self):
        self.resample_type = self._strategy.search_dict["resample_type"]
        if self.resample_type == 'list':
            self.resamples = self._strategy.search_dict["resample"]
        elif self.resample_type == 'arange_minutes':
            self.resamples = [f'{int(m)}min' for m in np.arange(*self._strategy.search_dict["resample"])]
        elif self.resample_type == 'empty':
            self.resamples = ["unresampled"]
        else:
            raise ParamError('Resample type error')

    @staticmethod
    def validate_search_params(params, types):
        implemented_types = ['fixed', 'list', 'arange', 'linspace', 'categorical', 'uniform']
        if len(params.keys()) != len(types.keys()):
            raise ParamError('Param and param type length mismatch in config')

        if set(params.keys()) != set(params.keys()):
            raise ParamError('Param and param type name mismatch in config')

        if not all([t in implemented_types for t in types.values()]):
            raise ParamError('Not all param types are implemented.')
        # any other search param validation logic here


class RandomSearcher(BaseParamSearcher):
   
    def __init__(self, strat_analyzer, data, train_percent, random_state):
        super().__init__(strat_analyzer, data, train_percent)
        # validation
        self.validate_search_params(self.prep_data_params, self.prep_data_params_types)
        self.validate_search_params(self.trading_data_params, self.trading_data_params_types)

        os.makedirs(search_path := f"{self._strategy.data_path}/random_search", exist_ok=True)
        self.random_state = random_state
        self.output_path = f"{search_path}/{self._strategy.name}_rs{self.random_state}.csv"
        self.headers += ['random_state']

    def search(self, n_iter):
        #  n_iter is the number of iteration PER RESAMPLE PERIOD
        self.init_writer(path=self.output_path)
        self.write(self.headers)

        for resample_p in self.resamples:
            if resample_p != 'unresampled':
                data = BackTester.resample_data(self.data, resample_p, self.prep_data_params)
            else:
                data = self.data

            # sample the params
            prep_params_sampled, trade_params_sampled = RandomSearcher.make_param_sample(
                self.prep_data_params,
                self.trading_data_params,
                self.prep_data_params_types,
                self.trading_data_params_types,
                n_iter=n_iter,
                random_state=self.random_state
            )

            for prep_params, trade_params in tqdm(zip(prep_params_sampled, trade_params_sampled)):
                data = self._strategy.prepare_data(data, prep_params)
                self._strategy.data = data
                self._strategy.set_trading_param(trade_params)
                self._strategy.backtest()
                self._portfolio_df = self.strat_analyzer.create_portfolio_return(
                    self._strategy.data_df,
                    "open",
                    "open",
                    self._strategy.trade_signals,
                    self._strategy.layer,
                    self._portfolio_params,
                    self._strategy.tp_sl_price_storage,
                )
                portfolio_summary, key_portfolio_summary, trade_dist, rolling_sharpe, _ = self.strat_analyzer.stats_calculate(self._portfolio_df, self._strategy.trade_signals)
                self.write(
                    [
                        resample_p,
                        *prep_params.values(),
                        *trade_params.values(),
                        portfolio_summary["sharpe"],
                        portfolio_summary["sortino"],
                        portfolio_summary["annualized_returns"],
                        portfolio_summary["trade_per_day"],
                        portfolio_summary["total_trades_number"],
                        -portfolio_summary["max_drawdown"],
                        portfolio_summary["ann_return/MDD"],
                        portfolio_summary["win_rate"],
                        portfolio_summary["win_loss_ratio"]
                    ]
                )
        self.close_writer()

    @staticmethod
    def make_param_sample(prep_params, trade_params, prep_params_types, trade_params_types, n_iter, random_state):

        np.random.seed(random_state)
        prep_params_candidates = []
        trade_params_candidates = []
        ref_list = []

        for ref in range(n_iter):
            for candidates, pt, pt_type in zip(
                    [prep_params_candidates, trade_params_candidates],
                    [prep_params, trade_params],
                    [prep_params_types, trade_params_types]
            ):
                param_choices = {}
                for field, ptype in pt_type.items():
                    if ptype == 'list':
                        choice = np.random.choice(pt[field])
                    elif ptype == 'fixed':
                        choice = np.random.choice(pt[field])
                    elif ptype == 'categorical':
                        choice = np.random.choice(pt[field])
                    elif ptype == 'arange':
                        choice = np.random.choice(np.arange(*pt[field]))
                    elif ptype == 'linspace':
                        choice = np.random.choice(np.linspace(*pt[field]))
                    elif ptype == 'uniform':
                        choice = np.random.uniform(*pt[field])
                    elif ptype == 'normal':
                        choice = np.random.normal(*pt[field])
                    elif ptype == 'lognormal':
                        choice = np.random.lognormal(*pt[field])
                    else:
                        raise NotImplementedError
                    param_choices[field] = choice
                candidates.append(param_choices)
            ref_list.append(ref)

        return prep_params_candidates, trade_params_candidates, ref_list


class RandomSearcherWithValidation(BaseParamSearcher):

    def __init__(self, strat_analyzer, train_data, test_data, random_state, spot_data=None, all_data=None):
        super().__init__(strat_analyzer, train_data=train_data, test_data=test_data, spot_data=spot_data, all_data=all_data)
        # use spot data or others data for do the validation
        self.validate_search_params(self.prep_data_params, self.prep_data_params_types)
        self.validate_search_params(self.trading_data_params, self.trading_data_params_types)

        os.makedirs(search_path := f"{self._strategy.data_path}/random_search", exist_ok=True)
        self.random_state = random_state
        self.output_path = f"{search_path}/{self._strategy.name}_rs{self.random_state}.csv"

        if spot_data is None:
            result_fields = ["sharpe", "annualized_returns", "trade_per_day", "max_drawdown", "Ann_PnL/MDD", "total_win_rate"]
            self.headers = (["resample"]
                            + list(self.prep_data_params.keys()) + list(self.trading_data_params.keys())
                            + [f'{field}_train' for field in result_fields] + [f'{field}_test' for field in result_fields]
                            )
        elif spot_data is not None:
            result_fields = ["sharpe", "annualized_returns", "trade_per_day", "max_drawdown", "Ann_PnL/MDD", "total_win_rate"]
            self.headers = (["resample"]
                            + list(self.prep_data_params.keys()) + list(self.trading_data_params.keys())
                            + [f'{field}_spot' for field in result_fields] + [f'{field}_train' for field in result_fields] + [f'{field}_test' for field in result_fields]
                            )

    def search(self, n_iter, use_spot=False, save_local=True, save_icloud=False, save_onedrive=False):
        self.init_writer(path=self.output_path)
        self.write(self.headers)

        for resample_p in self.resamples:
            spot_data = None
            if resample_p != 'unresampled':
                train_data = BackTester.resample_data(self.train, resample_p, self.prep_data_params)
                test_data = BackTester.resample_data(self.test, resample_p, self.prep_data_params)
                if self.spot is not None:
                    spot_data = BackTester.resample_data(self.spot, resample_p, self.prep_data_params)
            else:
                train_data = self.train
                test_data = self.test
                if self.spot is not None:
                    spot_data = self.spot

            # sample the params
            prep_params_sampled, trade_params_sampled, ref_list = RandomSearcher.make_param_sample(
                self.prep_data_params,
                self.trading_data_params,
                self.prep_data_params_types,
                self.trading_data_params_types,
                n_iter=n_iter,
                random_state=self.random_state
            )

            for ref, prep_params, trade_params in tqdm(zip(ref_list, prep_params_sampled, trade_params_sampled)):
                train_data = self._strategy.prepare_data(train_data, prep_params)
                test_data = self._strategy.prepare_data(test_data, prep_params)

                if spot_data is not None:
                    spot_data = self._strategy.prepare_data(spot_data, prep_params)
                    # spot data (validate data) data backtest
                    self._strategy.data = spot_data
                    self._strategy.set_trading_param(trade_params)
                    self._strategy.backtest()
                    self._portfolio_df = self.strat_analyzer.create_portfolio_return(
                        self._strategy.data_df,
                        "open",
                        "open",
                        self._strategy.trade_signals,
                        self._strategy.layer,
                        self._portfolio_params,
                        self._strategy.tp_sl_price_storage,
                    )
                    portfolio_summary_spot, _, _, _, _ = self.strat_analyzer.stats_calculate(self._portfolio_df, self._strategy.trade_signals)
                    
                # train data backtest
                self._strategy.data = train_data
                self._strategy.set_trading_param(trade_params)
                self._strategy.backtest()
                self._portfolio_df = self.strat_analyzer.create_portfolio_return(
                    self._strategy.data_df,
                    "open",
                    "open",
                    self._strategy.trade_signals,
                    self._strategy.layer,
                    self._portfolio_params,
                    self._strategy.tp_sl_price_storage,
                )
                portfolio_summary_train, _, _, _, _ = self.strat_analyzer.stats_calculate(self._portfolio_df, self._strategy.trade_signals)

                # test data backtests
                self._strategy.data = test_data
                self._strategy.set_trading_param(trade_params)
                self._strategy.backtest()
                self._portfolio_df = self.strat_analyzer.create_portfolio_return(
                    self._strategy.data_df,
                    "open",
                    "open",
                    self._strategy.trade_signals,
                    self._strategy.layer,
                    self._portfolio_params,
                    self._strategy.tp_sl_price_storage,
                )
                portfolio_summary_test, _, _, _, _ = self.strat_analyzer.stats_calculate(self._portfolio_df, self._strategy.trade_signals)
                
                # all data backtests
                self._strategy.set_trading_param(trade_params)
                self._strategy.backtest()
                self._portfolio_df = self.strat_analyzer.create_portfolio_return(
                    self._strategy.data_df,
                    "open",
                    "open",
                    self._strategy.trade_signals,
                    self._strategy.layer,
                    self._portfolio_params,
                    self._strategy.tp_sl_price_storage,
                )
                _, _, _, _, day_return_all = self.strat_analyzer.stats_calculate(self._portfolio_df, self._strategy.trade_signals)

                local_path = config["paths"]['local_directory']
                # drive_path_1 = config["paths"]['drive_directory_1']
                # drive_path_2 = config["paths"]['drive_directory_2']
                
                if save_local:
                    os.makedirs(local_path + f"/{self.random_state}", exist_ok=True)
                    day_return_all.to_csv(local_path + f'/{self.random_state}/{ref}_day_return.csv')

                # if save_icloud:
                #     os.makedirs(drive_path_1 + f"/{self.random_state}", exist_ok=True)
                #     day_return_all.to_csv(drive_path_1 + f'/{self.random_state}/{ref}_day_return.csv')

                # if save_onedrive:
                #     os.makedirs(drive_path_2 + f"/{self.random_state}", exist_ok=True)
                #     day_return_all.to_csv(drive_path_2 + f'/{self.random_state}/{ref}_day_return.csv')

                row_to_write = [resample_p, *prep_params.values(), *trade_params.values()]
                if spot_data is not None:
                    # add spot performance
                    row_to_write.extend([portfolio_summary_spot["sharpe"], portfolio_summary_spot["annualized_returns"], portfolio_summary_spot["trade_per_day"],
                                         - portfolio_summary_spot["max_drawdown"], portfolio_summary_spot["ann_return/MDD"], portfolio_summary_train["total_win_rate"]])

                row_to_write.extend([portfolio_summary_train["sharpe"], portfolio_summary_train["annualized_returns"], portfolio_summary_train["trade_per_day"],
                                     - portfolio_summary_train["max_drawdown"], portfolio_summary_train["ann_return/MDD"], portfolio_summary_train["total_win_rate"],

                                     portfolio_summary_test["sharpe"], portfolio_summary_test["annualized_returns"], portfolio_summary_test["trade_per_day"],
                                     - portfolio_summary_test["max_drawdown"], portfolio_summary_test["ann_return/MDD"], portfolio_summary_test["total_win_rate"]
                                     ])
                self.write(row_to_write)
        self.close_writer()

class OptunaSearcher(BaseParamSearcher):

    def __init__(self, strat_analyzer, data, train_percent, objective, method, random_state):
        super().__init__(strat_analyzer, data, train_percent)
        os.makedirs(search_path := f"{self._strategy.data_path}/optuna_search/{method}", exist_ok=True)
        self.random_state = random_state
        self.output_path = f"{search_path}/{self._strategy.name}_rs{self.random_state}.csv"
        self.headers += ['random_state', 'generation']
        self.psr = None
        self.objective = objective

        self._last_prep_params = None

    def _sharpe_objective_legacy(self, trial, prep_params, trade_params, prep_params_types, trade_params_types):
        '''
        MAXIMIZE sharpe as the direction is 'maximize' as argument passed in optuna.create_study()
        '''
        prep_params, trade_params = self._make_optuna_params(trial, prep_params, trade_params, prep_params_types, trade_params_types)
        data = self._strategy.prepare_data(self.resampled_data, prep_params)
        train = self._strategy.prepare_data(self.resampled_train, prep_params)
        test = self._strategy.prepare_data(self.resampled_test, prep_params)
        # self._strategy.data = data
        # do the train set first
        self._strategy.data = train
        self._strategy.set_trading_param(trade_params)
        self._strategy.backtest()  # for train
        self._portfolio_df = self.strat_analyzer.create_portfolio_return(
            self._strategy.data_df,
            "open",
            "open",
            self._strategy.trade_signals,
            self._strategy.layer,
            self._portfolio_params,
            self._strategy.tp_sl_price_storage,
            **self._portfolio_params,
        )
        portfolio_summary, key_portfolio_summary, _, _, _ = self.strat_analyzer.stats_calculate(self._portfolio_df, self._strategy.trade_signals, show=False)

        # set trial user attr for later output
        trial.set_user_attr('return', self._portfolio_df["cum_return"][-1])
        for attr, value in prep_params.items():
            trial.set_user_attr(attr, value)
        for attr, value in trade_params.items():
            trial.set_user_attr(attr, value)
        pf_summary_attrs = ['sharpe', 'ann_return/MDD', 'max_drawdown', 'win_rate', 'return_per_trade', 'trade_per_day']
        for pf_attr in pf_summary_attrs:
            trial.set_user_attr(pf_attr, portfolio_summary[pf_attr])

        sharpe_train = portfolio_summary["sharpe"] if portfolio_summary["sharpe"] is not None else 0

        # now, do the backtest and get sharpe and sortino for test set
        # the order of conducting train and test and store into attributes are important here
        self._strategy.data = test
        self._strategy.backtest()  # for test
        self._portfolio_df = self.strat_analyzer.create_portfolio_return(
            self._strategy.data_df,
            "open",
            "open",
            self._strategy.trade_signals,
            self._strategy.layer,
            self._portfolio_params,
            self._strategy.tp_sl_price_storage,
            **self._portfolio_params,
        )
        portfolio_summary, key_portfolio_summary, _, _, _ = self.strat_analyzer.cal_stats_from_daily_nav(
            self._portfolio_df, self._strategy.trade_signals, show=False
        )
        pf_summary_attrs_test = ['sharpe', 'sortino']
        for pf_attr in pf_summary_attrs_test:
            trial.set_user_attr(f'{pf_attr}_test', portfolio_summary[pf_attr])

        return sharpe_train

    def _sortino_objective(self, trial, prep_params, trade_params, prep_params_types, trade_params_types, funding_rates_series):

        prep_params, trade_params = self._make_optuna_params(trial, prep_params, trade_params, prep_params_types, trade_params_types)

        if self._last_prep_params != prep_params:
            # reduce time of preparing data
            data = self._strategy.prepare_data(self.resampled_data, prep_params)
            self._strategy.data = data
        self._strategy.set_trading_param(trade_params)
        self._strategy.backtest()

        # 1.01 is to cut out first 1% in validation set to avoid leakage
        if self._strategy.trade_signals.shape[0] >= 1:
            self._strategy.trade_signals_train = np.array([self._strategy.trade_signals[0][:int(len(self._strategy.data) * self.train_percent)]])
            self._strategy.trade_signals_test = np.array([self._strategy.trade_signals[0][int(len(self._strategy.data) * self.train_percent * 1.01):]])
        else:
            self._strategy.trade_signals_train = self._strategy.trade_signals[:int(len(self._strategy.data) * self.train_percent)]
            self._strategy.trade_signals_test = self._strategy.trade_signals[int(len(self._strategy.data) * self.train_percent * 1.01):]

        self._strategy.data_df_train = self._strategy.data_df.iloc[:int(len(self._strategy.data_df) * self.train_percent)]
        self._strategy.data_df_test = self._strategy.data_df.iloc[int(len(self._strategy.data_df) * self.train_percent * 1.01):]

        # add back funding rates into data_df
        if funding_rates_series is None:
            self._strategy.data_df_train['funding'] = np.zeros(len(self._strategy.data_df_train))
            self._strategy.data_df_test['funding'] = np.zeros(len(self._strategy.data_df_test))
        else:
            self._strategy.data_df_train = self._strategy.data_df_train.join(funding_rates_series, how='left').fillna(0)
            self._strategy.data_df_test = self._strategy.data_df_test.join(funding_rates_series, how='left').fillna(0)

        additional_args = {}
        if "portfolio_trade_method" in self._strategy.config.config:
            additional_args["portfolio_trade_method"] = self._strategy.config.config["portfolio_trade_method"]
            additional_args["weighting"] = self._strategy.config.config["weighting"]
            additional_args["bm_px_field"] = self.strat_analyzer.bm_px_field

        # create portfolio return for train and test separately
        self._portfolio_df_train = self.strat_analyzer.create_portfolio_return(
            self._strategy.data_df_train,
            "open",
            "open",
            self._strategy.trade_signals_train,
            self._strategy.layer,
            self._portfolio_params,
            self._strategy.tp_sl_price_storage,
            **additional_args
        )

        self._portfolio_df_test = self.strat_analyzer.create_portfolio_return(
            self._strategy.data_df_test,
            "open",
            "open",
            self._strategy.trade_signals_test,
            self._strategy.layer,
            self._portfolio_params,
            self._strategy.tp_sl_price_storage,
            **additional_args
        )

        # cal stats for train and test separately
        portfolio_summary_train, _, _, _, _ = self.strat_analyzer.cal_stats_from_daily_nav(self._portfolio_df_train, self._strategy.trade_signals_train)
        portfolio_summary_test, _, _, _, _ = self.strat_analyzer.cal_stats_from_daily_nav(self._portfolio_df_test, self._strategy.trade_signals_test)

        # set trial user attr for later output
        trial.set_user_attr('return', self._portfolio_df_train["cum_return"][-1])
        for attr, value in prep_params.items():
            trial.set_user_attr(attr, value)
        for attr, value in trade_params.items():
            trial.set_user_attr(attr, value)
        pf_summary_attrs_train = ['sharpe', 'sortino', 'l_s_ratio', 'holding_tm', 'return_per_trade', 'trade_per_day']
        for pf_attr in pf_summary_attrs_train:
            trial.set_user_attr(pf_attr, portfolio_summary_train[pf_attr])

        # set deflated sharpe
        if np.isfinite(sharpe := portfolio_summary_train['sharpe']):
            deflated_sharpe = self.psr.deflated_sharpe(idx=trial.number, result=sharpe)
        trial.set_user_attr('deflated_sharpe', deflated_sharpe)

        sortino_train = portfolio_summary_train["sortino"] if portfolio_summary_train["sortino"] is not None else 0
        pf_summary_attrs_test = ['sharpe', 'sortino']

        for pf_attr in pf_summary_attrs_test:
            trial.set_user_attr(f'{pf_attr}_test', portfolio_summary_test[pf_attr])

        self._last_prep_params = prep_params

        return sortino_train









