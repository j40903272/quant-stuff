import csv
import itertools
import os
import json

import hiplot as hip
# import numba as nb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest_engine.backtest_core.simulate_position import (
    simulate_const_weight_portfolio_trade_position,
    simulate_portfolio_trade_position, simulate_single_inst_position)

from numpy import arange
from scipy import stats
from scipy.ndimage.interpolation import shift
from statsmodels import regression
from tabulate import tabulate

from backtest_engine.funding_rates import FundingRates
from backtest_engine.plotter import Plotter
from backtest_engine.exchange_enum import *
from backtest_engine.utils import get_timeframe_string
from backtest_engine.path_handler import *
from backtest_engine.tunning_algo import *

CONSTANT_WEIGHTING = 'constant_weighting'

class StrategyAnalyzer:
    def __init__(self, strategy=None, portfolio_params=None, output_path=None, output_file_suffix=None):
        self._strategy = strategy
        self._plotter = Plotter(strategy)
        self._data_df = strategy.data_df
        self._symbols = [self._strategy.config.config["symbol"]] if isinstance(self._strategy.config.config["symbol"], str) else self._strategy.config.config["symbol"]
        if strategy is not None:
            self._portfolio_params = {}
            for symbol in self._symbols:
                self._portfolio_params[symbol] = StrategySettings.get_pf_param(
                    symbol=symbol,
                    exchange=self._strategy.config.config.get("exchange"),
                    position_scale=self._strategy.config.config.get("position_scale"),
                    compound = self._strategy.config.config.get("trading_data_param", {}).get("compound")
                )

            if isinstance(self._strategy.config.config["symbol"], list):
                for symbol in self._strategy.config.config["symbol"]:
                    self.enrich_funding_rates(self.get_funding_rates_series(symbol), symbol_name=symbol)
            else:
                self.enrich_funding_rates(self.get_funding_rates_series(self._strategy.config.config["symbol"]))
            self._output_path = f"{os.path.dirname(__file__)}/backtest_result" if output_path is None else output_path
            self._data_center_path = data_center_path
            self._output_file_suffix = output_file_suffix
            os.makedirs(self._output_path + "/html_report", exist_ok=True)
            os.makedirs(self._output_path + "/strategy_params", exist_ok=True)
        else:
            self._portfolio_params = portfolio_params
        self._strategy_df = pd.DataFrame()

    @property
    def data_df(self):
        return self._data_df

    @property
    def strategy_df(self):
        return self._strategy_df

    @property
    def symbols(self):
        return self._symbols

    @property
    def portfolio_params(self):
        return self._portfolio_params

    @property
    def strategy(self):
        return self._strategy

    @property
    def plotter(self):
        return self._plotter

    def prepare_data(self, data, prep_params):
        data = self._strategy.prepare_data(data, prep_params)
        self._strategy.data = data
        self._data_df = self._strategy.data_df
        if 'funding' in data.columns:
            self.enrich_funding_rates(data['funding'])
        else:
            if isinstance(self._strategy.config.config["symbol"], list):
                for symbol in self._strategy.config.config["symbol"]:
                    self.enrich_funding_rates(self.get_funding_rates_series(symbol), symbol_name=symbol)
            else:
                self.enrich_funding_rates(self.get_funding_rates_series(self._strategy.config.config["symbol"]))
        return data

    @staticmethod
    def simulate_position(np_bid, np_ask, trade_signals, no_of_layer, position_scale="base_value", capital=10000):

        current_position = 0
        current_layer = 0
        place_order = 0

        px = np.zeros(len(trade_signals))
        position = np.zeros(len(trade_signals))
        layers = np.zeros(len(trade_signals))
        # cash = np.zeros(len(trade_signals))

        np_mid = (np_bid + np_ask) / 2

        for idx in range(len(trade_signals)):
            if current_position != 0:
                px[idx] = np_mid[idx]
            if place_order > 0:
                current_layer += place_order
                # for transaction cost calculation (long, short same fee)
                # current_cash = current_cash - abs(place_order) * np_mid[idx] - transaction_cost * abs(place_order) * (np_mid[idx])
                if position_scale == "proportion":
                    current_position = capital * current_layer / no_of_layer / np_ask[idx]
                elif position_scale == "base_value":
                    if current_layer == 0:
                        current_position = 0
                    else:
                        current_position += capital * place_order / no_of_layer / np_ask[idx]
                else:
                    current_position += place_order
                px[idx] = np_ask[idx]  # sym1_bid
                place_order = 0
            elif place_order < 0:
                current_layer += place_order
                # current_cash = current_cash + abs(place_order) * np_mid[idx] - transaction_cost * abs(place_order) * (np_mid[idx])
                if position_scale == "proportion":
                    current_position = capital * current_layer / no_of_layer / np_bid[idx]
                elif position_scale == "base_value":
                    if current_layer == 0:
                        current_position = 0
                    else:
                        current_position += capital * place_order / no_of_layer / np_bid[idx]
                else:
                    current_position += place_order
                px[idx] = np_bid[idx]  # sym1_ask
                place_order = 0
            # cash[idx] = current_cash
            position[idx] = current_position

            if trade_signals[idx] != 0:
                place_order = trade_signals[idx]
            # position is made after the decision
            layers[idx] = current_layer

        return px, position, layers

    @classmethod
    def create_portfolio_return(
            cls,
            data,
            bid_field,
            ask_field,
            trade_signals,
            no_of_layer,
            pf_param_dict: dict,
            tp_sl_price_storage: dict,
            fx=None,
            bm_px_field="open",
            portfolio_trade_method=None,
            weighting=None,
    ):
        if fx is not None:
            if isinstance(fx, str):
                # it is col name of fx
                fx_value = data[fx]
            else:
                fx_value = fx
        else:
            fx_value = 1

        ref_pf_param = list(pf_param_dict.values())[0]
        if len(pf_param_dict) == 1:

            if 'tick_size' not in ref_pf_param:
                np_bid = (data[bid_field] * fx_value).to_numpy(dtype=np.float64)
                np_ask = (data[ask_field] * fx_value).to_numpy(dtype=np.float64)
            else:
                tick_size = ref_pf_param['tick_size']
                np_bid = ((data[bid_field] - tick_size / 2) * fx_value).to_numpy(dtype=np.float64)
                np_ask = ((data[ask_field] + tick_size / 2) * fx_value).to_numpy(dtype=np.float64)

            param = {}
            if 'position_scale' in ref_pf_param:
                param['position_scale'] = ref_pf_param['position_scale']
            if 'capital' in ref_pf_param:
                param['capital'] = ref_pf_param['capital']
            if 'compound' in ref_pf_param:
                param['compound'] = ref_pf_param['compound']
            px, position, layers = simulate_single_inst_position(np_bid, np_ask, trade_signals[0], no_of_layer, tp_sl_price_storage, **param)
            # funding rate suspend!!
            # try:
            #     funding_rates = data['funding'].to_numpy(dtype=np.float64)
            # except: Exception
            funding_rates = None
        else:
            bid_fields = [sym + '_' + bid_field for sym in pf_param_dict.keys()]
            ask_fields = [sym + '_' + ask_field for sym in pf_param_dict.keys()]

            if 'tick_size' not in list(pf_param_dict.values())[0]:
                np_bid = (data[bid_fields] * fx_value).to_numpy(dtype=np.float64)
                np_ask = (data[ask_fields] * fx_value).to_numpy(dtype=np.float64)
            else:
                tick_size_list = [i['tick_size'] / 2 for i in pf_param_dict.values()]
                np_bid = ((data[bid_fields] - tick_size_list) * fx_value).to_numpy(dtype=np.float64)
                np_ask = ((data[ask_fields] + tick_size_list) * fx_value).to_numpy(dtype=np.float64)

            if portfolio_trade_method == CONSTANT_WEIGHTING:
                px, position, layers = simulate_const_weight_portfolio_trade_position(
                    np_bid, 
                    np_ask, 
                    trade_signals[0], 
                    no_of_layer, 
                    ref_pf_param['position_scale'],                                       
                    ref_pf_param['capital'], weighting)
                funding_rates = None
            else:
                px, position, layers = simulate_portfolio_trade_position(np_bid, np_ask, trade_signals, no_of_layer, ref_pf_param['position_scale'], ref_pf_param['capital'])

        bm_px = data[bm_px_field].to_numpy(dtype=np.float64)
        capital = sum([i['capital'] for i in pf_param_dict.values()])
        portfolio = StrategyAnalyzer.create_portfolio_return_by_position(
            data,
            bm_px,
            position,
            px,
            layers,
            funding_rates,
            transaction_cost=ref_pf_param['transaction_cost'],
            capital=capital,
            multiplier=ref_pf_param['multiplier'],
            trans_cost_method=ref_pf_param['trans_cost_method'],
            overnight_cost_ratio=ref_pf_param['overnight_cost_ratio'],
            bm_px_field=bm_px_field,
            syms=list(pf_param_dict.keys())
        )
        del fx_value, np_bid, np_ask, px, position, layers, bm_px

        return portfolio

    # @staticmethod
    # def create_portfolio_return_by_position_legacy(
    #         data,
    #         bm_px,
    #         position,
    #         price_series,
    #         layer_series,
    #         funding_rates,
    #         transaction_cost=0.0003,
    #         capital=50000,
    #         multiplier=1,
    #         trans_cost_method="ratio",
    #         overnight_cost_ratio=None,
    #         bm_px_field="open",
    # ):
    #     portfolio = pd.DataFrame(index=data.index)
    #     portfolio["position"] = position
    #     portfolio["price"] = price_series
    #     portfolio["layers"] = layer_series

    #     portfolio['pnl'] = portfolio['position'] * portfolio['price'].diff().shift(-1) * multiplier

    #     portfolio["pnl"] = portfolio["pnl"] - (
    #             abs(portfolio["position"].diff()) * (portfolio["price"] * multiplier * transaction_cost if trans_cost_method == "ratio" else transaction_cost)).fillna(0)

    #     # overnight cost
    #     if overnight_cost_ratio is not None:
    #         portfolio["hour"] = portfolio.index.hour
    #         portfolio.loc[portfolio["hour"] == 16, "trade_date"] = portfolio.loc[portfolio["hour"] == 16].index.date
    #         portfolio["trade_date"].fillna(method="bfill", inplace=True)

    #         # fill last trade date of backtest period
    #         portfolio.loc[portfolio["trade_date"].isna(), "trade_date"] = portfolio.loc[portfolio["trade_date"].isna()].index.date

    #         need_overnight_cost = (portfolio["trade_date"].shift(-1) != portfolio["trade_date"]) & (portfolio["position"] != 0)

    #         portfolio.loc[need_overnight_cost, "pnl"] -= (abs(portfolio.loc[need_overnight_cost]["position"])
    #                                                       * (portfolio.loc[need_overnight_cost]["ref_price"] * multiplier * overnight_cost_ratio) / 365)

    #     portfolio["cum_pnl"] = portfolio["pnl"].cumsum()
    #     portfolio["cum_return"] = portfolio["cum_pnl"] / capital
    #     portfolio["return"] = portfolio["pnl"] / capital
    #     portfolio["bm_px"] = bm_px
    #     portfolio["bm_return"] = data[bm_px_field].diff() / data[bm_px_field].iloc[0]
    #     portfolio["return"] = portfolio["return"].fillna(0)
    #     portfolio["pnl"] = portfolio["pnl"].fillna(0)
    #     portfolio[["cum_pnl", "cum_return"]] = portfolio[["cum_pnl", "cum_return"]].fillna(method="ffill")

    #     return portfolio

    @staticmethod
    def create_portfolio_return_by_position(
            data,
            bm_px,
            position,
            price_series,
            layer_series,
            funding_rates,
            transaction_cost=0.0004,
            capital=50000,
            multiplier=1,
            trans_cost_method="ratio",
            overnight_cost_ratio=None,
            bm_px_field="open",
            syms=None,
    ):
        price_series = data[bm_px_field]
        if len(position.shape) > 1:
            pnl_unadjusted = position * np.diff(price_series, append=np.NaN, axis=0) * multiplier
            # transaction cost method: 'fix' or 'ratio'
            # real pnl is pnl minus transaction cost minus/plus funding fee if it is perp
            pnl = np.nan_to_num(
                pnl_unadjusted
                - np.nan_to_num(
                    abs(np.diff(position, prepend=np.NaN, axis=0)) * (price_series * multiplier * transaction_cost if trans_cost_method == "ratio" else transaction_cost))
                - np.nan_to_num(price_series * position * funding_rates)  # when position and funding rate signs are same, you have to pay
            )
        else:
            pnl_unadjusted = position * np.diff(price_series, append=np.NaN) * multiplier

            if type(funding_rates) is not None and type(funding_rates) == float:
                pnl = np.nan_to_num(
                    pnl_unadjusted
                    - np.nan_to_num(abs(np.diff(position, prepend=np.NaN)) * (price_series * multiplier * transaction_cost if trans_cost_method == "ratio" else transaction_cost))
                    - np.nan_to_num(price_series * position * funding_rates)  # when position and funding rate signs are same, you have to pay
                )

            else:
                # We usually get here! Due to there is no funding rates.
                pnl = np.nan_to_num(
                    pnl_unadjusted
                    - np.nan_to_num(abs(np.diff(position, prepend=np.NaN)) * (price_series * multiplier * transaction_cost if trans_cost_method == "ratio" else transaction_cost)) 
                )

        pnl_unadjusted = np.round(shift(pnl_unadjusted, 1), 6)
        pnl = np.round(shift(pnl, 1), 6)
        cum_pnl = np.cumsum(pnl, axis=0)
        cum_return = cum_pnl / capital
        return_ = np.nan_to_num(pnl / capital)
        bm_return = np.diff(bm_px, prepend=np.NaN) / bm_px[0]

        if len(position.shape) > 1:
            portfolio_dict = {}

            total_cum_pnl = np.transpose(np.sum(cum_pnl, axis=1))
            total_cum_return = np.transpose(total_cum_pnl / capital)
            total_return = np.transpose(np.sum(pnl, axis=1) / capital)

            position = np.transpose(position)
            funding_rates = np.transpose(funding_rates)
            price_series = np.transpose(price_series)
            layer_series = np.transpose(layer_series)
            pnl = np.transpose(pnl)
            pnl_unadjusted = np.transpose(pnl_unadjusted)
            cum_pnl = np.transpose(cum_pnl)
            cum_return = np.transpose(cum_return)
            return_ = np.transpose(return_)

            for i, sym in enumerate(syms):
                portfolio_dict[sym + '_position'] = position[i]
                portfolio_dict[sym + '_price'] = price_series[i]
                portfolio_dict[sym + '_layers'] = layer_series[i]
                portfolio_dict[sym + '_pnl'] = pnl[i]
                portfolio_dict[sym + '_pnl_unadjusted'] = pnl_unadjusted[i]
                portfolio_dict[sym + '_cum_pnl'] = cum_pnl[i]
                portfolio_dict[sym + '_cum_return'] = cum_return[i]
                portfolio_dict[sym + '_return'] = return_[i]

                if funding_rates.shape[0] == 1:
                    portfolio_dict[sym + '_funding_rate'] = funding_rates[0]  # case  when funding rates is only dummy ( zero series )
                else:
                    portfolio_dict[sym + '_funding_rate'] = funding_rates[i]
            portfolio_dict['layers'] = layer_series[0]
            portfolio_dict['position'] = position[0]  # set first for position ( only for publishing stat )
            portfolio_dict['cum_pnl'] = total_cum_pnl
            portfolio_dict['return'] = total_return
            portfolio_dict['cum_return'] = total_cum_return
        else:
            portfolio_dict = {
                'position': position,
                'funding_rate': funding_rates,
                'price': price_series,
                'ref_price': data[bm_px_field],
                'layers': layer_series,
                'pnl_unadjusted': pnl_unadjusted,
                'pnl': pnl,
                'cum_pnl': cum_pnl,
                'cum_return': cum_return,
                'return': return_,
                'transaction_cost': transaction_cost,
            }

        portfolio_dict['bm_return'] = bm_return
        return pd.DataFrame(portfolio_dict, index=data.index)

    @staticmethod
    def get_daily_return(portfolio, return_field, method="cumsum", filter_no_trade_day=False):
        # to get the dates which have trades
        if method == "cumsum":
            day_return = pd.DataFrame((portfolio[return_field]).resample("1D").sum())
        elif method == "cumprod":
            day_return = pd.DataFrame((portfolio[return_field] + 1).resample("1D").prod() - 1)

        # filter out the dates which have no trades
        if filter_no_trade_day:
            mean_return = (portfolio[return_field] + 1).resample("1D").mean()
            day_return = day_return.loc[mean_return[~mean_return.isna()].index]

        return day_return

    @staticmethod
    def get_sharpe_and_return(strategy, return_field="return", method="cumsum", filter_no_trade_day=False):

        def cal_risk_free_sharpe(day_return, return_field):
            std = (day_return[return_field] - 0.000035).std()
            return (day_return[return_field] - 0.000035).mean() / std * np.sqrt(252) if std != 0 else 0

        day_return = StrategyAnalyzer.get_daily_return(strategy, return_field)
        raw_sharpe = {"portfolio": StrategyAnalyzer.cal_raw_sharpe(day_return, return_field)}
        risk_free_sharpe = {
            "portfolio": cal_risk_free_sharpe(day_return, return_field)
        }  # risk free rate
        if method == "cumsum":
            day_return["cum_return"] = (day_return[return_field]).cumsum()
        elif method == "cumprod":
            day_return["cum_return"] = (day_return[return_field] + 1).cumprod()

        # pair trading in future 
        if "pnl1" in strategy.columns:
            day_return["pnl1"] = strategy["pnl1"].resample("1D").sum()
            day_return["cum_pnl1"] = day_return["pnl1"].cumsum()
            day_return1 = StrategyAnalyzer.get_daily_return(strategy, "return1")
            raw_sharpe["1"] = StrategyAnalyzer.cal_raw_sharpe(day_return1, "return1")
            risk_free_sharpe["1"] = cal_risk_free_sharpe(day_return1, "return1")
            day_return["return1"] = day_return1["return1"]

            day_return["pnl2"] = strategy["pnl2"].resample("1D").sum()
            day_return["cum_pnl2"] = day_return["pnl2"].cumsum()
            day_return2 = StrategyAnalyzer.get_daily_return(strategy, "return2")
            raw_sharpe["2"] = StrategyAnalyzer.cal_raw_sharpe(day_return2, "return2")
            risk_free_sharpe["2"] = cal_risk_free_sharpe(day_return2, "return2")
            day_return["return2"] = day_return2["return2"]

        return day_return, raw_sharpe, risk_free_sharpe

    @staticmethod
    def cal_raw_sharpe(day_return, return_field=None, sample_multiplier=None):
        if sample_multiplier is None:
            sample_length = day_return.index[1] - day_return.index[0]
            sample_multiplier = (60 * 60 * 24 * 365) / sample_length.total_seconds()

        if isinstance(day_return, pd.Series):
            return day_return.mean() / day_return.std() * np.sqrt(sample_multiplier) if day_return.mean() != 0 else 0
        else:
            return day_return[return_field].mean() / day_return[return_field].std() * np.sqrt(sample_multiplier) if day_return[return_field].mean() != 0 else 0

    def benchmarking(
            self,
            day_return,
            return_field="return",
            benchmark_col="bm_return",
            method="cumsum",
            filter_no_trade_day=False,
            trade_dist=None,
    ):
        benchmark_metrics = {}
        benchmark_metrics['bm_cum_return'] = None
        if benchmark_col is not None:

            def linreg(x, y):
                x = sm.add_constant(x.to_numpy())
                model = regression.linear_model.OLS(y, x).fit()
                return model.params[0], model.params[1]

            # the benchmark return col is derived from simple return
            benchmark_return = pd.DataFrame((self._strategy_df[benchmark_col]).resample("1D").sum())
            benchmark_metrics['alpha'], benchmark_metrics['beta'] = linreg(benchmark_return[benchmark_col], day_return[return_field])

            if method == "cumsum":
                benchmark_metrics['bm_cum_return'] = benchmark_return[benchmark_col].cumsum()
            elif method == "cumprod":
                benchmark_metrics['bm_cum_return'] = (benchmark_return[benchmark_col] + 1).cumprod()
        return benchmark_metrics

    def backtest(self):
        self._strategy.backtest()
        self._strategy_df = StrategyAnalyzer.create_portfolio_return(
            self._data_df, 
            "open", 
            "open", 
            self._strategy.trade_signals, 
            self._strategy.layer, 
            self._portfolio_params, 
            self._strategy.tp_sl_price_storage)

    def performance_analysis(self, save_result=True, show_plots=True, plot_bm=True):
        self.backtest()
        strategy_summary, key_strategy_summary, trade_dist, rolling_sharpe, day_return = StrategyAnalyzer.stats_calculate(self._strategy_df, self._strategy.trade_signals)
        benchmark_metrics = self.benchmarking(day_return)

        benchmark = benchmark_metrics['bm_cum_return'] if plot_bm else None
        self._plotter.plot_metrics(self._strategy_df, day_return=day_return, benchmark=benchmark, trade_dist=trade_dist, show=show_plots)
        if save_result:
            fout = open(f'{self._output_path}/{self._get_output_file_name()}.csv', "w", newline="")
            writer = csv.writer(fout)
            writer.writerow(
                [*self._strategy.config.prepare_data_param.values(), *self._strategy.config.trading_data_param.values(), self._strategy_df["cum_return"][-1],
                 *strategy_summary.values(), ]
            )
        return day_return, strategy_summary, trade_dist, rolling_sharpe, benchmark_metrics

    def backtest_analysis_report(
        self, 
        save_result=False, 
        print_result=True, 
        show_plots=True, 
        y_log_scale=True):

        self.backtest()
        strategy_summary, key_strategy_summary, trade_dist, rolling_sharpe, day_return = StrategyAnalyzer.stats_calculate(self._strategy_df, self._strategy.trade_signals, trade_details=self._strategy.trade_details)
        benchmark_metrics = self.benchmarking(day_return)

        strategy_summary_df = pd.DataFrame({'value': [k for k in strategy_summary.values()]}, index=[k for k in strategy_summary.keys()])
        if print_result:
            print("|-------------------------------------------|")
            print("|                  KEY Stats                |")
            print("|-------------------------------------------|")
            key_stats = pd.DataFrame({'value': [k for k in key_strategy_summary.values()]}, index=[k for k in key_strategy_summary.keys()])
            for i in key_stats.index:
                if isinstance((key_stats.loc[i])['value'], float) == True:
                    (key_stats.loc[i])['value'] = round((key_stats.loc[i])['value'],3)
            # print(key_stats)
            print(tabulate(key_stats, headers='keys', tablefmt='github'))
            print("|-------------------------------------------|")

        if show_plots or save_result:
            graph_objs = self._plotter.plot_metrics_interactive(
                self._strategy_df, 
                self._data_df, 
                day_return=day_return, 
                benchmark=benchmark_metrics['bm_cum_return'], 
                trade_dist=trade_dist, 
                strategy_summary_df=strategy_summary_df, 
                show=show_plots, 
                y_log_scale=y_log_scale)

        if save_result:
            # save to html in originl file
            with open(f"{self._output_path}/html_report/{self._get_output_file_name()}.html", 'w') as f:
                html_str = f"""
                    <div style="font-family: monospace, monospace;">
                        <p>{self._strategy.name}</p>
                    </div>
                """
                f.write(html_str)
                for obj in graph_objs:
                    f.write(obj.to_html(full_html=False, include_plotlyjs='cdn'))

            os.makedirs(f'{self._data_center_path}/engine_result/html_report', exist_ok=True)
            with open(f"{self._data_center_path}/engine_result/html_report/{self._get_output_file_name()}.html", 'w') as f:
                html_str = f"""
                    <div style="font-family: monospace, monospace;">
                        <p>{self._strategy.name}</p>
                    </div>
                """
                f.write(html_str)
                for obj in graph_objs:
                    f.write(obj.to_html(full_html=False, include_plotlyjs='cdn'))

            # print(f'Saved backtest result metrics to html: {self._output_path}/html_report/{self._get_output_file_name()}.html')

            # save param into csv
            fout = open(f'{self._output_path}/strategy_params/{self._get_output_file_name()}.csv', "w", newline="")
            writer = csv.writer(fout)
            writer.writerow(
                [
                    *self._strategy.config.prepare_data_param.values(),
                    *self._strategy.config.trading_data_param.values(),
                    self._strategy_df["cum_return"][-1],
                    *strategy_summary.values(),
                ]
            )
            # print(f'Saved backtest param to csv: {self._output_path}/strategy_params/{self._get_output_file_name()}.csv')
        return day_return, strategy_summary, trade_dist, rolling_sharpe, benchmark_metrics
    
    def random_search_with_validation(self, train_data, test_data, n_iter=10, random_state=42, spot_data=None, all_data=None , save_local=False, save_icloud=True, save_onedrive=True):
        RandomSearcherWithValidation(self, train_data, test_data, random_state, spot_data=spot_data, all_data=all_data).search(n_iter, save_local, save_icloud, save_onedrive)

    @staticmethod
    def stats_calculate(
        strategy_df, 
        trade_signals=None, 
        from_dt=None, 
        return_field="return", 
        method="cumsum", 
        filter_no_trade_day=False,
        trade_details=None):
    
        if "position" in strategy_df.columns:
            ##################
            #     LONG       #
            ##################

            # Date/Time at the moment of opening a long order
            open_long_index = strategy_df[(strategy_df['position'].shift(1) <= 0) & (strategy_df['position'] > 0)].index
            # Date/Time at the moment of closing a long order
            close_long_index = strategy_df[(strategy_df['position'].shift(1) > 0) & (strategy_df['position'] <= 0)].index
            if len(close_long_index) < len(open_long_index):
                for diff in range(len(close_long_index) - len(open_long_index), 0, 1):
                    open_long_index = open_long_index[:-1]

            # PnL at the moment of opening a long order
            open_long_cum_pnl = strategy_df.loc[open_long_index]['cum_pnl'].values  
            # Price at the moment of opening a long order
            open_long_price = strategy_df.loc[open_long_index]['ref_price'].values
            # Position at the moment of opening a long order
            open_long_position = strategy_df.loc[open_long_index]['position'].values

            # PnL at the moment of closing a long order
            close_long_cum_pnl = strategy_df.loc[close_long_index]['cum_pnl'].values
            # Price at the moment of closing a long order
            close_long_price = strategy_df.loc[close_long_index]['ref_price'].values
            # Position at the moment of closing a long order
            close_long_position = strategy_df.loc[close_long_index]['position'].values
            
            long_commission = abs((open_long_position - close_long_position) * (close_long_price) * strategy_df['transaction_cost'][0])

            gross_profit = 0
            gross_loss = 0
            # Orders that have not been close do not count, so here we use closed orders to calculate total trades 
            total_num_long_trades = len(open_long_index)

            long_trade_pnl = close_long_cum_pnl - open_long_cum_pnl - long_commission
            gross_profit += long_trade_pnl[long_trade_pnl > 0].sum() 
            gross_loss += long_trade_pnl[long_trade_pnl < 0].sum()
            num_long_winning_trades = np.count_nonzero(long_trade_pnl > 0)

            ##################
            #     Short      #
            ##################

            # Date/Time at the moment of opening a short order
            open_short_index = strategy_df[(strategy_df['position'].shift(1) >= 0) & (strategy_df['position'] < 0)].index
            # Date/Time at the moment of closing a short order
            close_short_index = strategy_df[(strategy_df['position'].shift(1) < 0) & (strategy_df['position'] >= 0)].index
            # Trigger the function when some orders have not been closed
            if len(close_short_index) < len(open_short_index):
                for diff in range(len(close_short_index) - len(open_short_index), 0, 1):
                    open_short_index = open_short_index[:-1]

            # PnL at the moment of opening a short order
            open_short_cum_pnl = strategy_df.loc[open_short_index]['cum_pnl'].values
            # Price at the moment of opening a short order
            open_short_price = strategy_df.loc[open_short_index]['ref_price'].values
            # Position at the moment of opening a short order
            open_short_position = strategy_df.loc[open_short_index]['position'].values

            # PnL at the moment of closing a short order
            close_short_cum_pnl = strategy_df.loc[close_short_index]['cum_pnl'].values
            # Price at the moment of closing a short order
            close_short_price = strategy_df.loc[close_short_index]['ref_price'].values
            # Position at the moment of closing a long order
            close_short_position = strategy_df.loc[close_short_index]['position'].values

            # Orders that have not been close do not count, so here we use closed orders to calculate total trades 
            total_num_short_trades = len(close_short_index)
            
            short_commission = abs((open_short_position - close_short_position) *  close_short_price * strategy_df['transaction_cost'][0])
            
            short_trade_pnl = close_short_cum_pnl - open_short_cum_pnl - short_commission
            gross_profit += short_trade_pnl[short_trade_pnl > 0].sum() 
            gross_loss += short_trade_pnl[short_trade_pnl < 0].sum()
            num_short_winning_trades = np.count_nonzero(short_trade_pnl > 0)
            
            ##################
            #     Overall    #
            ##################

            close_trade_cum_pnl = \
                strategy_df[
                    ((strategy_df['position'].shift(1) > 0) & (strategy_df['position'] <= 0)) | ((strategy_df['position'].shift(1) < 0) & (strategy_df['position'] >= 0))][
                    'cum_pnl'].values
            cum_pnl_diff = close_trade_cum_pnl[1:] - close_trade_cum_pnl[:-1]
            max_con_trade_losses = max([len(list(g)) for k, g in itertools.groupby(cum_pnl_diff < 0)], default=None)

            total_trade = total_num_long_trades + total_num_short_trades
            total_cum_pnl = sum(long_trade_pnl) + sum(short_trade_pnl)

            ##################
            #      Table     #
            ##################

            day_return, raw_sharpe, risk_free_sharpe = StrategyAnalyzer.get_sharpe_and_return(strategy_df, return_field, method, filter_no_trade_day)
            nav_from_one = day_return["cum_return"] + 1

            simple_ret = nav_from_one - nav_from_one.shift(1)
            simple_ret.iloc[0] = simple_ret.iloc[0] - 1
            simple_ret.fillna(0, inplace=True)

            if method == "cumsum":
                daily_ret = simple_ret
            elif method == "cumprod":
                daily_ret = nav_from_one / nav_from_one.shift(1) - 1
                daily_ret.iloc[0] = nav_from_one.iloc[0] / 1 - 1
                daily_ret.fillna(0, inplace=True)
            else:
                raise NotImplementedError

            daily_std = daily_ret.std()
            daily_downside_ret = daily_ret.copy()
            daily_downside_ret.loc[daily_downside_ret > 0] = 0
            daily_downside_std = daily_downside_ret.std()

            strategy_summary = {}
            strategy_summary['strategy_timeframe'] = get_timeframe_string(strategy_df)
            strategy_summary['raw_sharpe'] = raw_sharpe['portfolio']
            strategy_summary['risk_free_sharpe'] = risk_free_sharpe['portfolio']
            strategy_summary["gross_profit"] = gross_profit
            strategy_summary["gross_loss"] = gross_loss
            strategy_summary["net_profit"] = str(np.round(total_cum_pnl, 2)) + "$"
            strategy_summary["profit_factor"] = StrategyAnalyzer.to_none_if_denom_zero(-gross_profit, gross_loss)
            strategy_summary["win_days"] = len([x for x in daily_ret if x > 0])
            strategy_summary["lose_days"] = len([x for x in daily_ret if x < 0])
            strategy_summary["num_days"] = len(daily_ret)

            # the measurement of the cummulative return degree of linear 
            strategy_summary["max_daily_win"] = max([x for x in daily_ret if x > 0]) if strategy_summary["win_days"] > 0 else 0
            strategy_summary["max_daily_loss"] = min([x for x in daily_ret if x < 0]) if strategy_summary["lose_days"] > 0 else 0
            strategy_summary["avg_daily_win"] = strategy_summary["gross_profit"] / strategy_summary["win_days"] if strategy_summary["win_days"] > 0 else 0
            strategy_summary["avg_daily_loss"] = StrategyAnalyzer.to_none_if_denom_zero(strategy_summary["gross_loss"], strategy_summary["lose_days"])

            # MDD part!
            if method == 'cumsum':
                period_ret = nav_from_one.values[-1] - nav_from_one.values[0]
                drawdown = nav_from_one - nav_from_one.cummax()
            else:
                period_ret = nav_from_one.values[-1] / nav_from_one.values[0] - 1
                drawdown = nav_from_one / nav_from_one.cummax() - 1

            dd_add = drawdown.mean()
            backtest_length = nav_from_one.index[-1] - nav_from_one.index[0]
            ann_multiplier = (60 * 60 * 24 * 365) / backtest_length.total_seconds()

            strategy_summary["annualized_returns"] = period_ret * ann_multiplier
            strategy_summary["max_drawdown"] = drawdown.cummin().values[-1]
            strategy_summary["avg_drawdown"] = dd_add
            # strategy_summary["annualized_volatility"] = daily_std * np.sqrt(365)  # trade in 365

            sample_length = nav_from_one.index[1] - nav_from_one.index[0]
            sample_multiplier = (60 * 60 * 24 * 365) / sample_length.total_seconds()
            
            # ****************************** calculate sharpe *****************************
            trade_pnl = np.append(long_trade_pnl, short_trade_pnl)
            # strategy_summary["sharpe"] = StrategyAnalyzer.cal_raw_sharpe(daily_ret, sample_multiplier=sample_multiplier)
            expected_return = np.mean(trade_pnl)
            strategy_summary["sharpe"] = StrategyAnalyzer.to_none_if_denom_zero(expected_return - 0.2, np.std(trade_pnl, ddof=1)) # risk-free rate is fixed to 2% and it can only fit when capital 1000,
            target_return = 0
            downside_risk = np.sqrt(np.mean(np.minimum(0, trade_pnl - target_return) ** 2))
            strategy_summary["sortino"] = (expected_return - 0.2) / downside_risk
            
            # strategy_summary["sortino"] = StrategyAnalyzer.to_none_if_denom_zero(daily_ret.mean() * (sample_multiplier ** 0.5), daily_downside_std)
            # strategy_summary["SQN"] = daily_ret.mean() / daily_std * (len(nav_from_one) ** 0.5)
            strategy_summary["ann_return/MDD"] = StrategyAnalyzer.to_none_if_denom_zero(-strategy_summary["annualized_returns"], strategy_summary["max_drawdown"])
            strategy_summary["ann_return/ADD"] = StrategyAnalyzer.to_none_if_denom_zero(-strategy_summary["annualized_returns"], strategy_summary["avg_drawdown"])

            daily_port_ret = daily_ret.copy()
            daily_port_ret.sort_values(inplace=True, ascending=False)

            rolling_sharpe = day_return['return'].rolling(30).mean() / day_return['return'].rolling(30).std() * np.sqrt(sample_multiplier)
            # below is the old way to calculate rolling sharpe, which is very slow because of apply()
            # rolling_sharpe = day_return["return"].rolling(30).apply(lambda x: StrategyAnalyzer.cal_raw_sharpe(x, "return", sample_multiplier=sample_multiplier), raw=False)

            timeframe_sec = strategy_df.index.to_series().diff()[-1].total_seconds()

            shift_pos = strategy_df["position"].shift(1)
            long_no = len(strategy_df[strategy_df["position"] > shift_pos].loc[from_dt:])
            short_no = len(strategy_df[strategy_df["position"] < shift_pos].loc[from_dt:])
            strategy_summary['holding_ratio'] = (long_no + short_no) / len(strategy_df.loc[from_dt:])

            trade_pnl_df = pd.DataFrame({'Exit Data/Time': np.append(close_long_index, close_short_index)
                                         ,'Entry price': np.append(open_long_price, open_short_price)
                                         ,'Close price': np.append(close_long_price, close_short_price)
                                         ,'pnl': np.append(long_trade_pnl, short_trade_pnl)}
                                         ,index=np.append(open_long_index, open_short_index))
            trade_pnl_df.index.name= 'Entry Data/Time'
            strategy_summary["trade_pnl_df"] = trade_pnl_df

            # 下面部分要重寫
            strategy_summary['largest_winning_trade'] = np.round(trade_pnl_df["pnl"].max(), 2)
            strategy_summary['avg_winning_trade'] = np.round(trade_pnl_df[trade_pnl_df["pnl"] >= 0.0]["pnl"].mean(), 2)
            strategy_summary['largest_losing_trade'] = np.round(trade_pnl_df["pnl"].min(), 2)
            strategy_summary['avg_losing_trade'] = np.round(trade_pnl_df[trade_pnl_df["pnl"] < 0.0]["pnl"].mean(), 2)
            strategy_summary["win_loss_ratio"] = StrategyAnalyzer.to_none_if_denom_zero(-strategy_summary["avg_winning_trade"], strategy_summary["avg_losing_trade"])
            strategy_summary['number_winning_trade'] = num_long_winning_trades + num_short_winning_trades
            strategy_summary['number_losing_trade'] = total_trade - strategy_summary['number_winning_trade']
            strategy_summary['total_win_rate'] = (num_long_winning_trades + num_short_winning_trades) / (total_trade + 1e-8)
            strategy_summary['long_win_rate'] = num_long_winning_trades / (total_num_long_trades + 1e-8)
            strategy_summary['short_win_rate'] = num_short_winning_trades / (total_num_short_trades + 1e-8)
            
            
            if trade_details is not None:
                result_trade_details = {}
                for k, v in trade_details.items():
                    if 'time' in v:
                        v['time_in_hr'] = timeframe_sec * v['time'] / 3600
                        v.pop('time')
                    result_trade_details[strategy_df.index[k + 1]] = v

                df = pd.DataFrame.from_dict(result_trade_details, orient='index')
                strategy_summary["trade_pnl_df"] = trade_pnl_df.merge(df, left_index=True, right_index=True, how='outer')

        total_date = len(
            np.unique(((strategy_df.loc[from_dt:].index.to_numpy() - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "D")).astype(int)))

        trade_dist = None

        if trade_signals is not None:
            ref_signal = trade_signals[0]
            strategy_df["trade_signal"] = ref_signal[: len(strategy_df)]
            trade_dist = abs(strategy_df["trade_signal"]).resample("1D").sum()
            # strategy_summary['total_trade_number'] = total_trade = round(abs(ref_signal).sum() / 2 + 0.5)
            strategy_summary['total_trade_number'] = total_trade
            strategy_summary['long_trade_number'] = total_num_long_trades
            strategy_summary['short_trade_number'] = total_num_short_trades
            strategy_summary['holding_tm'] = len(strategy_df[strategy_df["position"] != 0]) / total_trade if total_trade != 0 else 0
            strategy_summary['avg_hour_holding'] = strategy_summary['holding_tm'] * timeframe_sec / 3600
            strategy_summary['return_per_trade'] = (strategy_df["cum_return"][-1] / total_trade if total_trade != 0 else 0)
            strategy_summary['pnl_per_trade'] = total_cum_pnl / total_trade if total_trade != 0 else 0
            strategy_summary["l_ratio"] = long_no / total_date if total_trade != 0 else 0
            strategy_summary["s_ratio"] = short_no / total_date if total_trade != 0 else 0
            strategy_summary["l_s_ratio"] = (strategy_summary['l_ratio'] + strategy_summary['s_ratio']) / 2
            strategy_summary["trade_per_day"] = trade_dist.mean()
            strategy_summary["day_for_a_trade"] = StrategyAnalyzer.to_none_if_denom_zero(1, trade_dist.mean())
            strategy_summary["max_continue_losses_trade_num"] = max_con_trade_losses

        strategy_summary["trade_pnl_df"].sort_index(inplace=True)
        # key portfolio summary
        key_strategy_summary = {k: v for k, v in strategy_summary.items() if k in [
            'strategy_timeframe', 
            'sharpe', 
            'annualized_returns', 
            'max_drawdown', 
            'ann_return/MDD',
            'ann_return',
            'avg_hour_holding', 
            'number_winning_trade',
            'number_losing_trade',
            'total_trade_number', 
            'total_win_rate', 
            'long_win_rate', 
            'short_win_rate',
            'long_trade_number', 
            'short_trade_number', 
            'sortino', 
            'net_profit',
            'gross_profit',
            'gross_loss',
            'profit_factor',
            'win_loss_ratio', 
            'annualized_volatility', 
            'pnl_per_trade', 
            'trade_per_day', 
            'day_for_a_trade',
            'max_continue_losses_trade_num', 
            'largest_winning_trade', 
            'winning_trade_pnl_median', 
            'avg_winning_trade', 
            'largest_losing_trade', 
            'losing_trade_pnl_median', 
            'avg_losing_trade']}

        return strategy_summary, key_strategy_summary, trade_dist, rolling_sharpe, day_return


    def plot_trading_signal(self, plot_fields=['open'], pos_anchor_field='open', from_=None, to_=None):
        self._plotter.plot_trade_signal(self._data_df, self._strategy_df, plot_fields, pos_anchor_field, from_=from_, to_=to_)

    def plot_trading_signal_interactive(self, plot_fields=['open'], pos_anchor_field='open', from_=None, to_=None, plot_subplot_fields=None):
        Plotter.plot_trade_signal_interactive(self._data_df, self._strategy_df, plot_fields, pos_anchor_field, from_=from_, to_=to_, plot_subplot_fields=plot_subplot_fields)

    def plot_return_series_interactive(self, day_return, benchmark_metrics, live_trade_return=None, live_trade_position_series=None, y_log_scale=False, show=True, from_=None,
                                       to_=None):
        o1, o2 = self._plotter.plot_return_series_interactive(
            day_return=day_return.loc[from_:to_], 
            portfolio=self.strategy_df.loc[from_:to_], 
            data_df=self.data_df.loc[from_:to_],            
            bm_cum_return=benchmark_metrics['bm_cum_return'], 
            show=show, 
            y_log_scale=y_log_scale,
            live_trade_return=live_trade_return, 
            live_trade_position_series=live_trade_position_series)

        return o1, o2

    def enrich_funding_rates(self, funding_rates: pd.Series, symbol_name=None):
        # symbol_name is only for multi instrument
        # if funding rates data not found, return zeroes in self.data length
        if funding_rates is None:
            if symbol_name is not None:
                field = symbol_name + '_funding'
            else:
                field = 'funding'
            self._data_df[field] = np.zeros(len(self._data_df))
        else:
            funding_rates = funding_rates.resample(self._strategy.resample_p, origin='epoch').sum().fillna(0)
            self._data_df = self._data_df.join(funding_rates, how='left').fillna(0)
            if symbol_name is not None:
                self._data_df.rename(columns={'funding': symbol_name + '_funding'}, inplace=True)

    @staticmethod
    def hiplot_analysis(path, file, rank_by=[]):
        data = pd.read_csv(f"{os.path.dirname(__file__)}/search_result/{path}/{file}")
        data = data[rank_by + [col for col in data.columns if col not in rank_by]]
        plot = hip.Experiment.from_dataframe(data)
        plot.display()

    @staticmethod
    def to_none_if_denom_zero(num, denom):
        if num is None or denom is None or denom == 0:
            return 0
        return num / denom

    @staticmethod
    def get_funding_rates_series(symbol) -> pd.Series:
        return FundingRates().get_rate(symbol)

    def _get_output_file_name(self):
        return self._strategy.name if self._output_file_suffix is None else self._strategy.name + '_' + self._output_file_suffix

    @staticmethod
    def cal_ew_portfolio_stats_from_daily_nav(all_return):

        # equal weight
        strategy_num = len(all_return.columns)
        weighting = 1 / strategy_num
        portfolio_all_return = pd.DataFrame(all_return.index).set_index('date')
        portfolio_all_return['pf_return'] = all_return.loc[:, all_return.columns != 'date'].sum(axis=1) * weighting
        portfolio_all_return['pf_nav'] = portfolio_all_return['pf_return'].cumsum()
        portfolio_all_return['pf_nav_from_1'] = portfolio_all_return['pf_nav'] + 1

        # portfolio stats table
        portfolio_stats = {}
        daily_return = portfolio_all_return['pf_return']
        daily_nav = portfolio_all_return['pf_nav']
        daily_std = daily_return.std()
        # downside return
        daily_downside_return = daily_return.copy()
        daily_downside_return.loc[daily_downside_return > 0] = 0
        daily_downside_std = daily_downside_return.std()
        nav_from_one = portfolio_all_return['pf_nav_from_1']
        period_return = nav_from_one.values[-1] / nav_from_one.values[0] - 1
        sample_length = daily_return.index[1] - daily_return.index[0]
        sample_multiplier = (60 * 60 * 24 * 365) / sample_length.total_seconds()
        total_length = daily_return.index[-1] - daily_return.index[0]
        ann_multiplier = (60 * 60 * 24 * 365) / total_length.total_seconds()

        daily_port_return = daily_return.copy()
        daily_port_return.sort_values(inplace=True, ascending=False)

        drawdown = nav_from_one / nav_from_one.cummax() - 1
        dd_add = drawdown.mean()
        win_days = len([x for x in daily_return if x > 0])
        lose_days = len([x for x in daily_return if x < 0])

        portfolio_stats['total_returns'] = daily_nav[-1]
        portfolio_stats["annualized_returns"] = period_return * ann_multiplier
        portfolio_stats['raw_sharpe'] = daily_return.mean() / daily_std * np.sqrt(sample_multiplier) if daily_return.mean() != 0 else 0
        portfolio_stats["gross_profit"] = sum([x for x in daily_return if x > 0])
        portfolio_stats["gross_loss"] = sum([x for x in daily_return if x < 0])
        portfolio_stats["net_profit"] = sum(daily_return)
        portfolio_stats["profit_factor"] = StrategyAnalyzer.to_none_if_denom_zero(-portfolio_stats["gross_profit"], portfolio_stats["gross_loss"])
        portfolio_stats["num_days"] = len(daily_return)
        portfolio_stats["win_days"] = len([x for x in daily_return if x > 0])
        portfolio_stats["lose_days"] = len([x for x in daily_return if x < 0])
        portfolio_stats["win_rate_daily"] = win_days / (win_days + lose_days + 1e-8)
        portfolio_stats["max_daily_win"] = max([x for x in daily_return if x > 0]) if portfolio_stats["win_days"] > 0 else 0
        portfolio_stats["max_daily_loss"] = min([x for x in daily_return if x < 0]) if portfolio_stats["lose_days"] > 0 else 0
        portfolio_stats["avg_daily_win"] = portfolio_stats["gross_profit"] / portfolio_stats["win_days"] if portfolio_stats["win_days"] > 0 else 0
        portfolio_stats["avg_daily_loss"] = StrategyAnalyzer.to_none_if_denom_zero(portfolio_stats["gross_loss"], portfolio_stats["lose_days"])
        portfolio_stats["win_loss_ratio"] = StrategyAnalyzer.to_none_if_denom_zero(-portfolio_stats["avg_daily_win"], portfolio_stats["avg_daily_loss"])

        # portfolio_stats["sharpe"] = cal_raw_sharpe(daily_return, sample_multiplier=sample_multiplier) 
        portfolio_stats["sortino"] = StrategyAnalyzer.to_none_if_denom_zero(daily_return.mean() * (sample_multiplier ** 0.5), daily_downside_std)
        portfolio_stats["max_drawdown"] = drawdown.cummin().values[-1]
        portfolio_stats["avg_drawdown"] = dd_add
        portfolio_stats["annualized_volatility"] = daily_std * np.sqrt(365)  # trade in 365
        portfolio_stats["ann_return/MDD"] = StrategyAnalyzer.to_none_if_denom_zero(-portfolio_stats["annualized_returns"], portfolio_stats["max_drawdown"])
        portfolio_stats["ann_return/ADD"] = StrategyAnalyzer.to_none_if_denom_zero(-portfolio_stats["annualized_returns"], portfolio_stats["avg_drawdown"])

        portfolio_stats = pd.DataFrame({'value': [f'{k:.4f}' for k in portfolio_stats.values()]}, index=[k for k in portfolio_stats.keys()])

        return portfolio_all_return, portfolio_stats
