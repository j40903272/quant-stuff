import numpy as np
import talib as ta
from backtest_engine.backtest_core.backtester import BackTester
from backtest_engine.bt_config import StrategyConfig


class SMAStrategyConfig(StrategyConfig):
    def __init__(self, strategy_class, json_config):
        super().__init__(strategy_class, json_config)

class SMAStrategy(BackTester):
    def __init__(self, json_config, data, prepare_data=True):
        self.config = SMAStrategyConfig(self.__class__.__name__, json_config)
        super().__init__(data, self.config, prepare_data=prepare_data)

    def set_trading_param(self, trading_data_param):
        self.SL_pct = trading_data_param["SL_pct"]
        self.TP_ratio = trading_data_param["TP_ratio"]

    def prepare_data(self, data, prepare_data_param):
        data['slow_sma'] = ta.SMA(data['close'], prepare_data_param['slow_sma_period'])
        data['fast_sma'] = ta.SMA(data['close'], prepare_data_param['fast_sma_period'])
        data['prev_slow_sma'] = data['slow_sma'].shift(1)
        data['prev_fast_sma'] = data['fast_sma'].shift(1)
        return data

    def run_logic(self, row):
        """
        0 "open"
        1 "high"
        2 "low"
        3 "close"
        4 "slow_sma"
        5 "fast_sma"
        6 "prev_slow_sma"
        7 "prev_fast_sma"
        """
        bullish = row[7] < row[6] and row[5] > row[4]
        bearish = row[7] > row[6] and row[5] < row[4]
        # print(bullish)
        # position status
        no_position = self.current_position == 0
        long_position = self.current_position > 0
        short_position = self.current_position < 0

        
        if long_position and bearish: # 1 -> 0
            self.sell_order_with_layer(directly_revert=False)
        elif short_position and bullish: # -1 -> 0
            self.buy_order_with_layer(directly_revert=False)
        elif no_position:
            if bullish: # 0 -> 1
                buy_sl = row[3] * (1 - self.SL_pct)
                buy_tp = row[3] * (1 + self.SL_pct * self.TP_ratio)
                # self.buy_order_with_layer(directly_revert=False, static_SL_price=buy_sl, static_TP_price=buy_tp)
                self.buy_order_with_layer(directly_revert=False)
            elif bearish: # 0 -> -1
                sell_sl = row[3] * (1 + self.SL_pct)
                sell_tp = row[3] * (1 - self.SL_pct * self.TP_ratio)
                # self.sell_order_with_layer(directly_revert=False, static_SL_price=sell_sl, static_TP_price=sell_tp)
                self.sell_order_with_layer(directly_revert=False)