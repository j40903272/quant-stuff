import pandas as pd
import numpy as np
cimport numpy as np
cimport cython
import ciso8601
import os

from backtest_engine.backtest_core.tp_param import StaticTakeProfitParam
from backtest_engine.backtest_core.sl_param import StaticStopLossParam


NO_SL = 0
SL = 1
ABSOLUTE_TRAILING_SL = 1
PERCENT_TRAILING_SL = 2
STATIC_PRICE_SL = 3

NO_TP = 0
STATIC_PRICE_TP = 1

cdef class BackTester(object):
    
    def __init__(self, data, config, start_date, end_date, trailing_stop_thres=0, prepare_data=True):

        filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
        start_index = (data.index >= start_date).argmax()
        end_index = (data.index <= end_date).argmin()

        self._data_path = os.path.dirname(__file__) + '/../search_result'
        self._name = config.config["name"]

        self._layer = config.trading_data_param['no_of_layer'] if 'no_of_layer' in config.trading_data_param else 1
        self._no_of_symbols =  1 if isinstance(config.config["symbol"],str) else len(config.config["symbol"])
        self._portfolio_trade_method = config.config["portfolio_trade_method"] if "portfolio_trade_method" in config.config else None

        self._buffer_period = config.trading_data_param['buffer_period'] if 'buffer_period' in config.trading_data_param else 0
        # self.target_fields = config.config["target_fields"]
        self._search_dict = config.config["search_param"] if "search_param" in config.config else {}

        self._set_sl_mode(config)
        self._set_tp_mode(config)

        self._resample_p = '1min'
        if prepare_data:
            self._resample_p = self.config.prepare_data_param['resample'] if 'resample' in self.config.prepare_data_param else self._resample_p
            if self._resample_p != '1min':
                data = self.resample_data(data, self._resample_p, self.config.prepare_data_param)
            data = self.prepare_data(data, self.config.prepare_data_param)

        
        # 在這邊決定data時間範圍應該就可以
        self._columns = data.columns
        self._trade_ref_price = 'open'
        self._trade_ref_high_price = 'high'
        self._trade_ref_low_price = 'low'
        self._data = data.values[start_index:end_index]
        
        self._dt_index = filtered_data.index.values
        self._start_dt = min(filtered_data.index.values)
        self._end_dt = max(filtered_data.index.values)

        self._open_pos_t = None
        self._trade_details = {}

        self.set_trading_param(self.config.trading_data_param)

        self._init()

    def _init(self):
        self._tp_sl_price_storage = {}

        self._trailing_stop_ref_px = 0
        self._current_position = 0
        self.avg_pf_value = 0
        self._buffer_period_cnt = 999999
        self._can_trade_flag = 0  # 0 means can trade
        self._layer_width = 0

        if self._portfolio_trade_method == 'constant_weighting':
            # constant weighting means it the hedge ratio between symbols won't change, so one trading signal is enough
            self._trade_signals = np.zeros((1, len(self._data)), dtype=int)
        else:
            self._trade_signals = np.zeros((self._no_of_symbols, len(self._data)), dtype=int)

        self._made_action = np.zeros(len(self._data), dtype=int)
        self.avg_pf_value_record = np.zeros(len(self._data))
        self._idx = 0
        self._row = np.zeros(1)


    @property
    def tp_sl_price_storage(self) -> dict:
        return self._tp_sl_price_storage

    @property
    def layer(self) -> int:
        return self._layer

    @property
    def search_dict(self) -> dict:
        return self._search_dict

    @property
    def data_path(self) -> str:
        return self._data_path

    @property
    def name(self) -> str:
        return self._name

    @property
    def buffer_period_cnt(self) -> int:
        return self._buffer_period_cnt
        
    @property
    def buffer_period(self) -> int:
        return self._buffer_period
       
    @property
    def current_position(self) -> int:
        return self._current_position
         
    @property
    def can_trade_flag(self) -> int:
        return self._can_trade_flag
        
    @property
    def trade_signals(self) -> np.ndarray:
        return self._trade_signals

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def data_df(self):
        # return pd.DataFrame(self._data, columns=self.target_fields, index=self._dt_index)
        return pd.DataFrame(self._data, columns=self._columns, index=self._dt_index)
    
    @property
    def resample_p(self):
        return self._resample_p
    
    @property
    def start_dt(self):
        return self._start_dt
    
    @property
    def end_dt(self):
        return self._end_dt

    @property
    def dt_index(self):
        return self._dt_index

    @property
    def no_of_symbols(self):
        return self._no_of_symbols

    @property
    def portfolio_trade_method(self):
        return self._portfolio_trade_method

    @property
    def trade_details(self):
        return self._trade_details

    @buffer_period_cnt.setter
    def buffer_period_cnt(self, buffer_period_cnt):
        self._buffer_period_cnt = buffer_period_cnt

    @buffer_period.setter
    def buffer_period(self, buffer_period):
        self._buffer_period = buffer_period

    @current_position.setter
    def current_position(self, current_position):
        self._current_position = current_position

    @can_trade_flag.setter
    def can_trade_flag(self, can_trade_flag):
        self._can_trade_flag = can_trade_flag

    @trade_signals.setter
    def trade_signals(self, trade_signals):
        self._trade_signals = trade_signals

    @data.setter
    def data(self, data_df):
        self._data = data_df.values
        self._dt_index = data_df.index.values

    def _set_sl_mode(self, config):
        if config.trading_data_param['SL_mode'] == 'no_sl':
            self._SL_mode = NO_SL
        else:
            self._SL_mode = SL

        self._active_sl_param = None
        self._inactive_sl_param = {}
        self._inactive_sl_trigger_prices = []
        self._need_update_ref_price = False

    def _set_tp_mode(self, config):
        if config.trading_data_param['TP_mode'] == 'no_tp':
            self._TP_mode = NO_TP
            self._tp_param = None
        elif config.trading_data_param['TP_mode'] == 'static_price':
            self._TP_mode = STATIC_PRICE_TP
            self._tp_param = None


    @staticmethod
    def resample_data(data, resample_p, prepare_data_params, end_bar_time=False):
        data_df = pd.DataFrame()
        for col in data.columns:
            if 'open' in col:
                data_df[col] = data[col].resample(resample_p, origin='epoch').first()
            elif 'close' in col:
                data_df[col] = data[col].resample(resample_p, origin='epoch').last()
        if 'high' in data.columns:
            data_df['high'] = data['high'].resample(resample_p, origin='epoch').max()
        if 'low' in data.columns:
            data_df['low'] = data['low'].resample(resample_p, origin='epoch').min()
        if 'bid' in data.columns:
            data_df['bid'] = data['bid'].resample(resample_p, origin='epoch').last()
        if 'ask' in data.columns:
            data_df['ask'] = data['ask'].resample(resample_p, origin='epoch').last()
        if 'volume' in data.columns:
            data_df['volume'] = data['volume'].resample(resample_p, origin='epoch').sum()
        if 'funding' in data.columns:
            data_df['funding'] = data['funding'].resample(resample_p, origin='epoch').sum()
        if 'agg_fields' in prepare_data_params:
            for field, method in prepare_data_params['agg_fields'].items():
                if method == 'sum':
                    data_df[field] = data[field].resample(resample_p, origin='epoch').sum()
                elif method == 'last':
                    data_df[field] = data[field].resample(resample_p, origin='epoch').last()
                elif method == 'max':
                    data_df[field] = data[field].resample(resample_p, origin='epoch').max()
                elif method == 'min':
                    data_df[field] = data[field].resample(resample_p, origin='epoch').min()

        if end_bar_time:
            data_df = data_df.shift(1)

        data_df = data_df.dropna()
        return data_df

    def prepare_data(self, data, prepare_data_param):
        pass

    def set_trading_param(self, trading_data_param):
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def backtest(self):
        self._init()
        for idx, row in enumerate(self._data):
            #print(idx,row)
            self._buffer_period_cnt += 1
            self._idx = idx
            # convert each row to a dictionary
            self._row = dict(zip(self._columns, row))
            self.run_logic(self._row)
            if self._SL_mode != NO_SL:
                self._stop_loss_process(idx)
            if self._TP_mode != NO_TP:
                self._take_profit_process(idx)

        return self._trade_signals, self.avg_pf_value_record

    def _get_long_take_profit_thres(self):
        return self._tp_param.next_tp_price

    def _get_short_take_profit_thres(self):
        return self._tp_param.next_tp_price

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _stop_loss_process(self, idx, sym_index = 0):
        # using the high or low price as reference for stop-loss triggers; using close price may not be a faithful representation of live trade but acceptable
        price = self._row[self._trade_ref_high_price] if self._current_position < 0 else self._row[self._trade_ref_low_price]
        for trigger_price in self._inactive_sl_trigger_prices:
            if self._current_position > 0 and price >= trigger_price:
                param = self._inactive_sl_param[trigger_price]
                if self._active_sl_param is None or param.get_stop_loss_price(price) > self._active_sl_param.get_stop_loss_price(price):
                    if param.SL_MODE == ABSOLUTE_TRAILING_SL or param.SL_MODE == PERCENT_TRAILING_SL:
                        self._need_update_ref_price = True
                        self._trailing_stop_ref_px = price
                    self._active_sl_param = param
                    self._inactive_sl_param.pop(trigger_price)
                    self._inactive_sl_trigger_prices.remove(trigger_price)
            elif self._current_position < 0 and price <= trigger_price:
                param = self._inactive_sl_param[trigger_price]
                if self._active_sl_param is None or param.get_stop_loss_price(price) < self._active_sl_param.get_stop_loss_price(price):
                    if param.SL_MODE == ABSOLUTE_TRAILING_SL or param.SL_MODE == PERCENT_TRAILING_SL:
                        self._need_update_ref_price = True
                        self._trailing_stop_ref_px = price
                    self._active_sl_param = param
                    self._inactive_sl_param.pop(trigger_price)
                    self._inactive_sl_trigger_prices.remove(trigger_price)


        if self._need_update_ref_price:
            # ABSOLUTE_TRAILING_Sl or PERCENT_TRAILING_SL
            if self._current_position > 0:
                self._trailing_stop_ref_px = max(self._trailing_stop_ref_px, price)
            elif self._current_position < 0:
                self._trailing_stop_ref_px = min(self._trailing_stop_ref_px, price)

        if self._trade_signals[sym_index][idx] == 0 and self._active_sl_param is not None:
            # dun have signal yet
            if self._current_position > 0 and price <= self._active_sl_param.get_stop_loss_price(self._trailing_stop_ref_px):
                self._tp_sl_price_storage[idx] = self._active_sl_param.get_stop_loss_price(self._trailing_stop_ref_px)
                self._close_position_process(idx, "stop loss")
            elif self._current_position < 0 and price >= self._active_sl_param.get_stop_loss_price(self._trailing_stop_ref_px):
                # print(f"{row[10]} - trailing stop {self._trailing_stop_ref_px}, , current px {price}  - {self._trade_signals[sym_index][idx]}")
                self._tp_sl_price_storage[idx] = self._active_sl_param.get_stop_loss_price(self._trailing_stop_ref_px)
                self._close_position_process(idx, "stop loss")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _take_profit_process(self, idx, sym_index = 0):
        if self._trade_signals[sym_index][idx] == 0 and self._tp_param is not None:
            # dun have signal yet
            # using the high or low price as reference for take-profit triggers; using close price may not be a faithful representation of live trade but acceptable
            price = self._row[self._trade_ref_high_price] if self._current_position > 0 else self._row[self._trade_ref_low_price]

            if self._current_position > 0 and price >= self._get_long_take_profit_thres():
                self._tp_sl_price_storage[idx] = self._get_long_take_profit_thres()
                if self._tp_param.get_next_tp_layer() < self._current_position:
                    # print(f'sell: {self._tp_param.get_next_tp_layer()} , {self._current_position}')
                    self.sell_order_with_layer(directly_revert=False, layer=self._tp_param.get_next_tp_layer(), action_type="take profit")
                    self._tp_param.hit_tp()
                else:
                    self._close_position_process(idx, "take profit")
            elif self._current_position < 0 and price <= self._get_short_take_profit_thres():
                # print(f"{row[10]} - take profit {self._tp_param}, , current px {price}  - {self._trade_signals[sym_index][idx]}")
                self._tp_sl_price_storage[idx] = self._get_short_take_profit_thres()
                if self._tp_param.get_next_tp_layer() < abs(self._current_position):
                    # print(f'buy: {self._tp_param.get_next_tp_layer()} , {self._current_position}')
                    self.buy_order_with_layer(directly_revert=False, layer=self._tp_param.get_next_tp_layer(), action_type="take profit")
                    self._tp_param.hit_tp()
                else:
                    self._close_position_process(idx, "take profit")



    @cython.boundscheck(False)
    @cython.wraparound(False)
    ## static_SL_price is not used when SL_mode is STATIC_PRICE_SL !!!
    ## static_TP_price is not used when TP_mode is STATIC_PRICE_TP !!!
    def buy_order_with_layer(self, directly_revert=True, static_SL_price = None, static_TP_price = None, layer = 1, tp_param = None, sl_param_list = None,
                             action_type = None):
        # print(self._layer, self._current_position, self._current_position < self._layer)
        price = self._row[self._trade_ref_price]
        if self._made_action[self._idx] != 0:
            # print(f" {self._idx} buy order -- {self._made_action[self._idx]} is set - {self._current_position}")
            return
        if self._buffer_period_cnt < self._buffer_period:
            return

        if self._current_position <= self._layer - layer :
            prev_position = self._current_position
            if 0 <= self._current_position or not directly_revert:
                self._current_position += layer
                self._trade_signals[0][self._idx] = layer
                self.avg_pf_value = (price + self.avg_pf_value * (abs(self._current_position) - 1)) / abs(self._current_position) if self._current_position != 0 else 0
            else:
                self._trade_signals[0][self._idx] = -self._current_position + layer
                self._current_position = layer
                self._clean_params(action_type)
                self.avg_pf_value = price

            if prev_position == 0 or (prev_position * self._current_position < 0):
                self._open_pos_t = self._idx

            self._buffer_period_cnt = 0  # after enter market, enter buffer period
            if static_SL_price is not None:
                self._trailing_stop_ref_px = static_SL_price
                self._active_sl_param = StaticStopLossParam(static_SL_price)
            if static_TP_price is not None:
                self._tp_param = StaticTakeProfitParam({static_TP_price: layer}, True)
            # print(f" {self._idx} buy order -- {self._trade_signals[0][self._idx]} at {price} - {self._current_position} {self._layer}")
            self._made_action[self._idx] = 1
            if tp_param is not None:
                self._tp_param = tp_param
            if sl_param_list is not None:
                self._set_sl_param(sl_param_list)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    ## static_SL_price is not used when SL_mode is STATIC_PRICE_SL !!!
    ## static_TP_price is not used when TP_mode is STATIC_PRICE_TP !!!
    def sell_order_with_layer(self, directly_revert=True, static_SL_price = None, static_TP_price = None, layer = 1, tp_param = None, sl_param_list = None,
                              action_type = None):
        price = self._row[self._trade_ref_price]
        if self._made_action[self._idx] != 0:
            # print(f" {self._idx} sell order -- {self._made_action[self._idx]} is set - {self._current_position}")
            return
        if self._buffer_period_cnt < self._buffer_period:
            return

        if self._current_position >= -self._layer + layer:
            prev_position = self._current_position
            if 0 >= self._current_position or not directly_revert:
                self._current_position -= layer
                self._trade_signals[0][self._idx] = -layer
                self.avg_pf_value = (price + self.avg_pf_value * (abs(self._current_position) - 1)) / abs(self._current_position) if self._current_position != 0 else 0
            else:
                self._trade_signals[0][self._idx] = -self._current_position - layer
                self._current_position = -layer
                self._clean_params(action_type)
                self.avg_pf_value = price

            if prev_position == 0 or (prev_position * self._current_position < 0):
                self._open_pos_t = self._idx

            self._buffer_period_cnt = 0  # after enter market, enter buffer period
            if static_SL_price is not None:
                self._trailing_stop_ref_px = static_SL_price
                self._active_sl_param = StaticStopLossParam(static_SL_price)
            if static_TP_price is not None:
                self._tp_param = StaticTakeProfitParam({static_TP_price: layer}, False)
            # print(f" {self._idx} sell order -- {self._trade_signals[0][self._idx]} at {price} - {self._current_position} {self._layer}")
            self._made_action[self._idx] = 1
            if tp_param is not None:
                self._tp_param = tp_param
            if sl_param_list is not None:
                self._set_sl_param(sl_param_list)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def buy_portfolio_order_with_layer(self, signal_weights, directly_revert=True):
        if self._made_action[self._idx] != 0:
            # print(f" {self._idx} buy order -- {self._made_action[self._idx]} is set - {self._current_position}")
            return
        if self._buffer_period_cnt < self._buffer_period:
            return

        if self._current_position < self._layer:
            if 0 <= self._current_position or not directly_revert:
                self._current_position += 1
                for i in range(len(signal_weights)):
                    self._trade_signals[i][self._idx] = signal_weights[i]
            else:
                for i in range(len(signal_weights)):
                    self._trade_signals[i][self._idx] = -self._current_position + signal_weights[i]
                self._current_position = 1
            self._buffer_period_cnt = 0  # after enter market, enter buffer period
            self._made_action[self._idx] = 1
            if self._current_position == 0:
                self._clean_params(None)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def sell_portfolio_order_with_layer(self, signal_weights, directly_revert=True):
        if self._made_action[self._idx] != 0:
            # print(f" {self._idx} sell order -- {self._made_action[self._idx]} is set - {self._current_position}")
            return
        if self._buffer_period_cnt < self._buffer_period:
            return

        if self._current_position > -self._layer:
            if 0 >= self._current_position or not directly_revert:
                self._current_position -= 1
                for i in range(len(signal_weights)):
                    self._trade_signals[i][self._idx] = signal_weights[i]
            else:
                for i in range(len(signal_weights)):
                    self._trade_signals[i][self._idx] = -self._current_position + signal_weights[i]
                self._current_position = -1
            self._buffer_period_cnt = 0  # after enter market, enter buffer period
            self._made_action[self._idx] = 1
            if self._current_position == 0:
                self._clean_params(None)

    def close_position(self, action_type=None):
        if self._made_action[self._idx] == 0:
            self._close_position_process(self._idx, action_type)

    def _close_position_process(self, idx, action_type):
        self._trade_signals[0][idx] = -self._current_position
        self._current_position = 0
        self._clean_params(action_type)

    def _clean_params(self, action_type):
        self._layer_width = 0
        self._avg_pf_value = 0
        self._buffer_period_cnt = 0  # after enter market, enter buffer period
        self._trailing_stop_ref_px = 0
        self._tp_param = None
        self._set_sl_param([])
        self._mark_trade_details(action_type)

    def _mark_trade_details(self, action_type):
        if self._open_pos_t != None:
            self._trade_details[self._open_pos_t] = {'time':self._idx - self._open_pos_t, 'action': action_type}

    def _set_sl_param(self, sl_param_list):
        self._inactive_sl_param = {}
        self._need_update_ref_price = False
        self._active_sl_param = None
        for sl_param in sl_param_list:
            if sl_param.SL_MODE == STATIC_PRICE_SL:
                self._active_sl_param = sl_param
            elif sl_param.SL_MODE == ABSOLUTE_TRAILING_SL or sl_param.SL_MODE == PERCENT_TRAILING_SL:
                if sl_param.active_directly:
                    # dun expect there is static SL param also in this case
                    self._active_sl_param = sl_param
                else:
                    self._inactive_sl_param[sl_param.trigger_price] = sl_param

        self._inactive_sl_trigger_prices = list(self._inactive_sl_param.keys())

    def close_portfolio_order_position(self):
        if self._made_action[self._idx] == 0:
            self._close_portfolio_order_position_process(self._idx)

    def _close_portfolio_order_position_process(self, idx):
        self._trade_signals[0][idx] = -self._current_position
        self._current_position = 0
        self._layer_width = 0
        self._avg_pf_value = 0
        self._buffer_period_cnt = 0  # after enter market, enter buffer period
        self._trailing_stop_ref_px = 0
        self._tp_param = None

    def run_logic(self, row):
        pass