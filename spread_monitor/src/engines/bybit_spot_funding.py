# %%z
import os
import sys
import math
import json
import time
import statistics
import datetime
from itertools import product
from unicodedata import category
import pandas as pd
import numpy as np

from ..utils import *


class Detector():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.coin_settle = self.exchange.settle
        self.coin_margin = self.exchange.margin
        # log
        self.log_path = r"log"
        self.logger = Logger(logger_folder=self.log_path,
                             file_name='{}'.format(self.exchange.name))
        self.refresh()
        self.exchange.set_logger(self.logger)

    def refresh(self):
        # init variables
        self.crit = self.cfg_signal['crit']
        self.reduce_crit = self.cfg_signal['reduce_crit']
        self.positions = {}
        self.margin_assets = {}
        self.future_order = {}
        self.spot_order = {}

        # update data
        self.exchange.set_info()
        self.load_datas()


    def get_spreads(self, symbol):
        _value = max(20, self.cfg_order['max_value'])
        _, _turnover = self.exchange.get_turnover24h(symbol)
        _value = min((_turnover / 24) *
                     self.cfg_order['max_value_1Hr_Vol_pct']/100, _value)
        _real_prices = {}
        for category, side in product(['spot', 'linear'], ['bids', 'asks']):
            _real_prices[f'{category}_{side}_price'] = get_real_price(
                self.exchange, category, symbol, _value, side)

        return _value, \
            (_real_prices['linear_bids_price'] / _real_prices['spot_bids_price']-1)*100, \
            (_real_prices['spot_asks_price'] /
             _real_prices['linear_asks_price']-1)*100
    


