import numpy as np
import talib as ta
from backtest_engine.backtest_core.backtester import BackTester
from backtest_engine.bt_config import StrategyConfig


class ForceStrategyConfig(StrategyConfig):
    def __init__(self, strategy_class, json_config):
        super().__init__(strategy_class, json_config)

class ForceStrategy(BackTester):
    def __init__(self, json_config, data, start_date, end_date, prepare_data=True):
        self.config = ForceStrategyConfig(self.__class__.__name__, json_config)
        super().__init__(data, self.config, start_date, end_date , prepare_data=prepare_data)

    def set_trading_param(self, trading_data_param):
        self.SL_pct = trading_data_param["SL_pct"]
        self.TP_ratio = trading_data_param["TP_ratio"]

    def prepare_data(self, data, prepare_data_param):
        volume_period = prepare_data_param['volume_period']
        data['slow_sma'] = ta.SMA(data['close'], prepare_data_param['slow_sma_period'])
        data['fast_sma'] = ta.SMA(data['close'], prepare_data_param['fast_sma_period'])
        data['prev_slow_sma'] = data['slow_sma'].shift(1)
        data['prev_fast_sma'] = data['fast_sma'].shift(1)
        data['priceV1hr'] = (data['close'] - data['close'].shift(1))/data['close']
        data['cumVolume'] = data['volume'].rolling(window = volume_period).sum()
        data['highWindow'] = data['high'].rolling(window = prepare_data_param['mass_period']).max()
        data['lowWindow'] = data['low'].rolling(window = prepare_data_param['mass_period']).min()
        data['upMass'] = data['close'] / data['highWindow']
        data['downMass'] = data['lowWindow'] / data['close']
        data['Vup'] = ( data['priceV1hr'] *data['cumVolume'] + data['volume'] / data['downMass'] )/ (1/data['downMass'] + 1/data['upMass'])
        data['Vdown'] = data['volume'] - data['Vup']
        data['MAForce'] = ta.EMA(data['Vup'] - data['Vdown'],prepare_data_param['force_period'])
        data['preMaForce'] = data['MAForce'].shift(1)
        data['volumeEma'] = ta.EMA(data['volume'],volume_period)
        data['volumeSma'] = ta.SMA(data['volume'],volume_period)
        data['Force'] = data['Vup'] - data['Vdown']
        data['closeForce'] = ta.EMA(data['Force'],prepare_data_param['close_force_period'])
        # print(data[20:40])
        return data

    def run_logic(self, row):
        """
        0 "MAForce"
        1 "preMaForce"
        2 "volume"
        3 "volumeEma"
        4 "closeForce"
        5 "cumVolume"
        6 "open"
        7 "close"
        8 "volumeSma"
        9 "high",
        10 "low"
        """
        bullish = row['MAForce'] > 0 and row['preMaForce'] < 0 and row['volume'] > row['volumeSma']
        bearish = row['closeForce'] < 0
        no_position = self.current_position == 0
        long_position = self.current_position > 0

        if long_position and bearish: # 1 -> 0
            self.sell_order_with_layer(directly_revert=False)
            # self.sell_order_with_layer(directly_revert=False, static_SL_price = 0.99, static_TP_price= 1.01)
        elif no_position:
            if bullish: # 0 -> 1
                self.buy_order_with_layer(directly_revert=False)
                # self.buy_order_with_layer(directly_revert=False, static_SL_price = 0.99, static_TP_price= 1.01)