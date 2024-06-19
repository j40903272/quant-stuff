import talib as ta
from backtest_engine.backtest_core.backtester import BackTester
from backtest_engine.bt_config import StrategyConfig


class EfficiencyRatioZscoreStrategyConfig(StrategyConfig):
    def __init__(self, strategy_class, json_config):
        super().__init__(strategy_class, json_config)


class EfficiencyRatioZscoreStrategy(BackTester):
    def __init__(self, json_config, data):
        self.config = EfficiencyRatioZscoreStrategyConfig(self.__class__.__name__, json_config)
        super().__init__(data, self.config)

    def set_trading_param(self, trading_data_param):
        self.SL_pct = trading_data_param["SL_pct"]
        self.TP_ratio = trading_data_param["TP_ratio"]

    def prepare_data(self, data, prepare_data_param):

        data["price_delta"] = (data["close"] - data["close"].shift(1))
        data["total_price_delta"] = (data["close"] - data["close"].shift(prepare_data_param["er_p"]))
        data["delta_abs_sum"] = ta.SUM(data["price_delta"].abs(), timeperiod=prepare_data_param["er_p"])

        data["efficiency_ratio"] = data["total_price_delta"]/data["delta_abs_sum"]
        data["prev_efficiency_ratio"] = data["efficiency_ratio"].shift(1)

        # zscore
        data["efficiency_ratio_avg"] = ta.SMA(data["efficiency_ratio"], timeperiod=prepare_data_param["zscore_p"])
        data["efficiency_ratio_stddev"] = ta.STDDEV(data["efficiency_ratio"], timeperiod=prepare_data_param["zscore_p"])
        data["efficiency_ratio_zscore"] = (data["prev_efficiency_ratio"] - data["efficiency_ratio_avg"])/data["efficiency_ratio_stddev"]
        data["prev_efficiency_ratio_zscore"] = data["efficiency_ratio_zscore"].shift(1)

        return data

    def run_logic(self, row):

        """
        0 "open",
        1 "high",
        2 "low",
        3 "close",
        4 "price_delta",
        5 "total_price_delta",
        6 "delta_abs_sum",
        7 "efficiency_ratio",
        8 "efficiency_ratio_avg",
        9 "efficiency_ratio_stddev",
        10 "efficiency_ratio_zscore",
        11 "prev_efficiency_ratio_zscore"

        03 |----------|
        02 |----------| 
        01 |----------| 
        00 |----------| 
        01 |----------| 
        02 |----------|
        03 |----------| 

        """

        CrossUp_00 = (row[10] >= 0.0) and (row[11]< 0.0)
        CrossUp_01 = (row[10] >= 1.0) and (row[11]< 1.0)
        CrossUp_02 = (row[10] >= 2.0) and (row[11]< 2.0)
        CrossUp_03 = (row[10] >= 3.0) and (row[11]< 3.0)

        CrossDown_00 = (row[10] <= 0.0) and (row[11]> 0.0)
        CrossDown_n01 = (row[10] <= -1.0) and (row[11]> -1.0)
        CrossDown_n02 = (row[10] <= -2.0) and (row[11]> -2.0)
        CrossDown_n03 = (row[10] <= -3.0) and (row[11]> -3.0)

        # position status
        no_position = self.current_position == 0
        long_position = self.current_position > 0
        short_position = self.current_position < 0

        # 1 -> 0
        if long_position and (CrossUp_03 or CrossDown_00):
            self.sell_order_with_layer(directly_revert=False)
        # -1 -> 0
        elif short_position and (CrossDown_n03 or CrossUp_00): 
            self.buy_order_with_layer(directly_revert=False)
        elif no_position:

            # 0 -> 1
            if CrossUp_01:

                buy_sl = row[3] * (1 - self.SL_pct)
                buy_tp = row[3] * (1 + self.SL_pct * self.TP_ratio)
                self.buy_order_with_layer(directly_revert=False, static_SL_price=buy_sl, static_TP_price=buy_tp)

            # 0 -> -1
            elif CrossDown_n01:

                sell_sl = row[3] * (1 + self.SL_pct)
                sell_tp = row[3] * (1 - self.SL_pct * self.TP_ratio)

                self.sell_order_with_layer(directly_revert=False, static_SL_price=sell_sl, static_TP_price=sell_tp)
