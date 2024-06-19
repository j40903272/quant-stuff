# %%
from .exchange_base import ExchangeBase
from ..sdks import *
import time
import pandas as pd
from datetime import datetime
import random
from decimal import *
import warnings

# %%


class Bybit(ExchangeBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rest = unified.HTTP(
            endpoint="https://api.bybit.com",
            api_key=self._api_key,
            api_secret=self._api_secret)

        self.ws_linear = unified.WebSocket(
            ws_name="USDT Perp",
            test=False,
            api_key=self._api_key,
            api_secret=self._api_secret,
            suffix='public/linear')

        # self.ws_spot = unified.WebSocket(
        #     ws_name="Spot",
        #     test=False,
        #     api_key=self._api_key,
        #     api_secret=self._api_secret,
        #     suffix='public/spot')

        # self.ws_pvt = unified.WebSocket(
        #     ws_name="Private",
        #     test=False,
        #     api_key=self._api_key,
        #     api_secret=self._api_secret,
        #     suffix='private')

        self.symbol_list = []

    def close_all_ws(self):
        self.ws_linear.ws.exit()
        self.ws_spot.ws.exit()
        self.ws_pvt.ws.exit()

    def set_info(self, advanced=True):
        self.set_symbol_list()
        self.set_account_info()
        self.filter_symbol()
        self.set_market_info()

        if advanced:
            self.set_funding_rates()

        self.set_ws_data()

    def set_symbol_list(self):
        req_status, rsp = self.send_request(self.rest.get_collateral_info)
        if req_status:
            for row in rsp['result']['list']:
                self.collateral_info[row['currency']] = float(
                    row['collateralRatio'])
                self.borrow_rates[row['currency']] = float(
                    row['hourlyBorrowRate']) * 24 * 36500
                # 'SHIB'
                if row['currency'] not in self.block_list and float(row['collateralRatio']) >= self.haricut_threshold and row['borrowable'] and row['marginCollateral']:
                    self.symbol_list.append(f"{row['currency']}{self.settle}")

        return req_status

    def set_borrow_rates(self):
        req_status, rsp = self.send_request(self.rest.get_collateral_info)
        if req_status:
            for row in rsp['result']['list']:
                self.borrow_rates[row['currency']] = float(
                    row['hourlyBorrowRate']) * 24 * 36500

        return req_status


    def filter_symbol(self):
        symbols = set(list(self.positions.keys()) +
                      [f'{coin}{self.settle}' for coin in self.balances.keys()])
        self.reduce_only_list = [
            symbol for symbol in symbols if symbol not in self.symbol_list]
        self.symbol_list = list(set(self.symbol_list + self.reduce_only_list))

    def set_account_info(self):
        req_status, rsp = self.send_request(
            self.rest.get_wallet_balance, accountType='UNIFIED')

        if req_status:
            account_info = rsp['result']['list'][0]
            self.total_equity = float(account_info['totalEquity'])
            self.wallet_usd = self.set_wallet_usd(account_info)
            self.ltv = float(account_info['accountLTV'])
            self.im_rate = float(account_info['accountIMRate'])
            self.mm_rate = float(account_info['accountMMRate'])
            self.upl = float(account_info['totalPerpUPL'])
            self.set_usdValue_dict(account_info['coin'], None)

        return req_status

    def refresh_account_info(self):
        # 以後需改成websocket
        return self.set_account_info()

    def get_mmr(self):
        return self.mm_rate

    def get_imr(self):
        return self.im_rate

    def get_upl(self):
        return self.upl

    def get_leverage(self):
        pos_val = sum([abs(self.usdValue_dict[k])
                      for k in self.usdValue_dict.keys() if len(k) > 4 and self.settle in k])
        return pos_val / self.wallet_usd

    # ws
    def set_ws_data(self):
        self.ws_pvt.subscribe_private_channels()
        for symbol in self.symbol_list:
            self.ws_linear.get_ticker(symbol)
            self.ws_linear.get_orderbook(symbol)
            self.ws_linear.get_trade(symbol)
            self.ws_spot.get_orderbook(symbol)
            self.ws_spot.get_ticker(symbol)
            self.ws_spot.get_trade(symbol)

    # markets

    def set_market_info(self):
        req_spot_status, spot_rsp = self.send_request(
            self.rest.info, category='spot')
        req_linear_status, linear_rsp = self.send_request(
            self.rest.info, category='linear')

        if _status := (req_spot_status and req_linear_status):
            # # linear
            for row in linear_rsp['result']['list']:
                if row['symbol'] in self.symbol_list and row['contractType'] == 'LinearPerpetual':
                    self.linear_info[row['symbol']] = {
                        'pricePrecision': float(row['priceFilter']['tickSize']),
                        'quantityPrecision': float(row['lotSizeFilter']['qtyStep']),
                        'minProvideSize': float(row['lotSizeFilter']['minOrderQty'])
                    }
                    if row['fundingInterval'] != 480:
                        self.special_symbols[row['symbol']
                                             ] = row['fundingInterval']

            # spot
            for row in spot_rsp['result']['list']:
                if row['symbol'] in self.symbol_list:
                    self.spot_info[row['symbol']] = {
                        'pricePrecision': float(row['priceFilter']['tickSize']),
                        'quantityPrecision': float(row['lotSizeFilter']['basePrecision']),
                        'minProvideSize': float(row['lotSizeFilter']['minOrderQty'])
                    }

        return _status

    def get_pricePrecision(self, symbol, category='spot'):
        return eval(f'self.{category}_info')[symbol]['pricePrecision']

    def get_quantityPrecision(self, symbol, category='spot'):
        return eval(f'self.{category}_info')[symbol]['quantityPrecision']

    def get_minProvideSize(self, symbol, category='spot'):
        return eval(f'self.{category}_info')[symbol]['minProvideSize']

    def get_sizeIncrement(self, symbol, category='spot'):
        return eval(f'self.{category}_info')[symbol]['quantityPrecision']

    def get_priceIncrement(self, symbol, category='spot'):
        return eval(f'self.{category}_info')[symbol]['pricePrecision']

    def get_orderbook(self, symbol, category='linear'):
        orderbook = eval(f'self.ws_{category}').get_orderbook(symbol)
        if(orderbook is None or len(orderbook) == 0):
            self.logger.info(f'Use restful to get orderbook {symbol}')
            try:
                rsp = self.rest.get_orderbook(
                    category=category, symbol=symbol, limit=50)
                # print(rsp)
                # orderbook = {
                #     'bids': [[float(elem[0]), float(elem[1])] for elem in rsp['result']['b']],
                #     'asks': [[float(elem[0]), float(elem[1])] for elem in rsp['result']['a']],
                # }
            except:
                time.sleep(0.05)
                rsp = self.rest.get_orderbook(
                    category=category, symbol=symbol, limit=50)
                # print(rsp)
            orderbook = {
                'bids': [[float(elem[0]), float(elem[1])] for elem in rsp['result']['b']],
                'asks': [[float(elem[0]), float(elem[1])] for elem in rsp['result']['a']],
            }
            ts = rsp['time']
        else:
            now = datetime.now()
            ts = int(now.timestamp()*1000)
        return orderbook, ts

    def get_trades(self, symbol, category='spot'):
        trades = eval(f'self.ws_{category}').get_trade(symbol)
        if(trades is None or len(trades) == 0):
            self.logger.info(f'Use restful to get trades {symbol}')
            try:
                rsp = self.rest.trades(
                    category=category, symbol=symbol)
            except:
                time.sleep(0.05)
                rsp = self.rest.trades(
                    category=category, symbol=symbol)
            trades = rsp['result']['list']
        return trades

    def get_last_fr(self, symbol):
        ticker = self.ws_linear.get_ticker(symbol)
        if(ticker is None):
            _status, ticker = self.send_request(
                self.rest.query_symbol, category='linear', symbol=symbol)
            if not _status:
                self.logger.error(f'Failed to get next fr for {symbol}')
                return _status, 0
            else:
                return _status, float(ticker['result']['list'][0]['fundingRate'])
        else:
            return True, float(ticker['fundingRate'])


    def get_price(self, symbol, category='linear'):
        if category == 'linear':
            ticker = self.ws_linear.get_ticker(symbol)
        else:
            ticker = self.ws_spot.get_ticker(symbol)
        if(ticker is None):
            _status, ticker = self.send_request(
                self.rest.query_symbol, category=category, symbol=symbol)
            if not _status:
                self.logger.error(f'Failed to get price for {symbol}')
                return _status, 0
            else:
                return _status, float(ticker['result']['list'][0]['lastPrice'])
        else:
            return True, float(ticker['lastPrice'])

# %%
