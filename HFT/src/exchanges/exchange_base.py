# %%
from abc import abstractmethod
from abc import ABCMeta
from abc import ABC
from tempfile import TemporaryFile
from unicodedata import category
import pandas as pd
import numpy as np
import statistics
from decimal import *
import warnings
from ..utils import Logger
import time

import datetime
warnings.filterwarnings("ignore")
# %%


class ExchangeBase(object, metaclass=ABCMeta):
    """
    交易所資訊：Account、FR、BR、price、orderbook、下單、抽單、改單
    """

    def __init__(self, **kwargs):
        """
            exchange='FTX', coin_margined='USD', _api_key='', _api_secret='', _subaccount=''
        """
        self.__dict__.update(kwargs)

        # user
        self.usdValue_dict = {}  # {'BTC':499,'BTC-PERP':-499}
        self.wallet_usd = 0.0
        self.leverage = 0.0

        # markets
        self.funding_rates = {}
        self.borrow_rates = {}

        # # log
        self.log_path = r"log"
        self.logger = Logger(logger_folder=self.log_path,
                             file_name='{}'.format(self.name))

    def send_request(self, func, **kwargs):
        try:
            rsp = func(**kwargs)
        except:
            time.sleep(0.02)
            self.logger.error('Send restful request failed')
            rsp = func(**kwargs)
        finally:
            if rsp['retMsg'] != 'OK':
                return False, rsp
            else:
                return True, rsp

    def set_logger(self, logger):
        self.logger = logger

    def get_symbols(self, coin_based='USDT'):
        return [symbol for symbol in self.market_info_table if coin_based in symbol]

    # 計算小數點後幾位
    def decimal_places(self, num):
        decimal_num = Decimal(str(num))
        return -decimal_num.as_tuple().exponent

    def get_enable_amount(self, amount, symbol, category):
        minProvideSize = float(self.get_minProvideSize(symbol, category))
        sizeIncrement = float(self.get_sizeIncrement(symbol, category))
        _size_increment = str(sizeIncrement)
        if(amount < minProvideSize):
            amount = 0
        else:
            amount = float(
                (Decimal(str(amount))//Decimal(_size_increment))*Decimal(_size_increment))

        try:
            return round(amount, self.decimal_places(self.get_quantityPrecision(symbol, category)))
        except:
            return amount

    def get_enable_price(self, price, symbol, category):
        priceIncrement = self.get_priceIncrement(symbol, category)
        priceIncrement = str(priceIncrement)
        price = float((Decimal(str(price))//Decimal(priceIncrement)) *
                      Decimal(priceIncrement))

        try:
            return round(price, self.decimal_places(self.get_pricePrecision(symbol, category)))
        except:
            return price

    def get_real_price(self, symbol, value, side, category):
        orderbook = self.get_orderbook(symbol, category)
        init_value = value
        total_amt = 0
        global_price = 0
        side = 'bids' if side == 'Sell' else 'asks'
        for order in orderbook[side]:
            price = order[0]
            amt = order[1]
            value_per_order = (price*amt)
            total_amt += amt
            value -= value_per_order
            global_price = price
            if(value < 0):
                break

        total_amt += (value/global_price)
        real_price = init_value/total_amt
        return real_price

    def get_spread(self, _buy_price, _sell_price):
        # _buy_price = self.get_real_price(_buy, _value, 'buy')
        # _sell_price = self.get_real_price(_sell, _value, 'sell')
        return (_sell_price/_buy_price-1)*100

    def get_usdValue(self, symbol):
        try:
            usdvalue = self.usdValue_dict[symbol]
        except:
            usdvalue = 0
        if(abs(usdvalue) < 1):
            usdvalue = 0
        return usdvalue

    def get_leverage(self):
        return self.leverage

    def get_wallet_usd(self):
        return self.wallet_usd

    def get_weight(self, symbol):
        return self.get_usdValue(symbol) / self.wallet_usd

    def calc_buffered_price(self, symbol, buffer, side, category):
        order_book = self.get_orderbook(symbol, category)
        _maker_price = order_book['bids'][0][0]/(
            1+buffer/100) if side == 'Buy' else order_book['asks'][0][0]*(1+buffer/100)
        return self.get_enable_price(_maker_price, symbol, category)

    def calc_mid_price(self, symbol, side, category, forced=False, lastest_price=0):
        order_book = self.get_orderbook(symbol, category)
        _op = -1 if side == 'Buy' else 1
        _ba_side = 'bids' if side == 'Sell' else 'asks'
        _price_increment = self.get_priceIncrement(symbol, category)
        price = order_book[_ba_side][0][0]

        if not forced and price == lastest_price:
            return price
        else:
            tick = (Decimal(
                order_book['asks'][0][0] - order_book['bids'][0][0]) / Decimal(_price_increment)) // 2
            if tick == 0:
                tick = 1
            price = float(Decimal(str(price)) + _op *
                          tick*Decimal(_price_increment))

            return self.get_enable_price(price, symbol, category)

    # abstract method
    # user

    def set_usdValue_dict(self):
        pass

    def set_leverage(self):
        pass

    def set_wallet_usd(self):
        pass

    # ws
    def set_ws_data(self):
        pass

    def ws_monitor(self):
        pass

    # markets
    def get_orderbook(self, symbol):
        pass

    def get_minProvideSize(self, symbol):
        pass

    def get_sizeIncrement(self, symbol):
        pass

    def get_priceIncrement(self, symbol):
        pass

    def get_minNotional(self, symbol):
        pass

    # trades
    def get_buffer(self, symbol, category, side='Buy', buffer_n=1000, buffer_q=0.99):
        # isBuyerMaker = True if side == 'BUT' else False
        df = pd.DataFrame(self.get_trades(symbol=symbol, category=category))
        try:
            df = df.rename(columns={'T': 'time', 'S': 'side', 'p': 'price'})
        except:
            pass

        df = df[df['side'] == side]
        first = df[['time', 'price']].groupby(
            'time').first().astype(float)['price']
        last = df[['time', 'price']].groupby(
            'time').last().astype(float)['price']
        spread = abs(first-last)*100/first
        buffer = spread.quantile(buffer_q)
        return buffer

    def get_maker_price(self, symbol, side, buffer_n, buffer_q, category):
        # self.get_orderbook(symbol)
        buffer = self.get_buffer(symbol, category, side, buffer_n, buffer_q)
        if side == 'Buy':
            price = float(self.get_orderbook(symbol, category)
                          ['bids'][0][0])/(1+buffer/100)
        else:
            price = float(self.get_orderbook(symbol, category)
                          ['asks'][0][0])*(1+buffer/100)

        return self.get_enable_price(symbol=symbol, price=price, category=category)

    def place_order(self, symbol, value, side, price, type, postonly):
        pass

    def modify_order(self):
        pass

    def cancel_order(self):
        pass

    def cancel_orders(self):
        pass

    def check_order(self):
        pass
