from tempfile import TemporaryFile
from ._http_manager import _UnifiedHTTPManager
from ._websocket_stream import _V5WebSocketManager
from ._websocket_stream import USDT_PERPETUAL
from ._websocket_stream import _identify_ws_method, _make_public_kwargs
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
import json
import re

ws_name = USDT_PERPETUAL


class HTTP(_UnifiedHTTPManager):
    def predicted_funding_rate(self, **kwargs):
        """
        Get predicted funding rate and my funding fee.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/linear/#t-predictedfunding.
        :returns: Request results as dictionary.
        """

        suffix = "/private/linear/funding/predicted-funding"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )


class WebSocket():
    def __init__(self, ws_name, **kwargs):
        # super().__init__(ws_name, **kwargs)

        self.private = False if 'private' not in kwargs['suffix'] else True
        self.kwargs = kwargs if self.private else _make_public_kwargs(kwargs)

        self.ws = _V5WebSocketManager(
            ws_name, **self.kwargs)
        self.ws._connect()

        '''新增變數'''
        self._subscriptions = {}
        self._orderbooks = {}
        self._orderbooks_timestamps = {}
        self._trades = {}
        self._tickers = {}

        self._position = {}
        self._fills = []
        self._orders = {}
        self._wallet = {}
        print('*'*10, f'reset {ws_name} ws', '*'*10)

    def _ws_ping(self):
        self.ws.ws.send(json.dumps({"op": "ping"}))

    def _ws_subscribe(self, topic, callback, subscription_args):
        if not self.ws:
            self.ws = _V5WebSocketManager(
                ws_name, **self.kwargs)
            self.ws._connect()
        self.ws.subscribe(topic, callback, subscription_args)
        self.ws.sync_subscriptions(self._subscriptions)

    def _unified(self, text):
        if re.match(r'[a-zA-Z0-9]{41,43}', text) is None:
            return str(text).upper()
        else:
            return str(text)

    def check_subscribed(self, topic, symbol):
        if topic not in self._subscriptions.keys():
            self._subscriptions[topic] = [symbol]
            return True
        else:
            if symbol in self._subscriptions[topic]:
                return False
            else:
                self._subscriptions[topic].append(symbol)
                return True

    def prepare_subscription_args(self, topic, list_of_symbols):
        """
        Prepares the topic for subscription by formatting it with the
        desired symbols.
        """
        if topic in self.ws.private_topics:
            # private topics do not support filters
            return [topic]
        else:
            return [f'{topic}.{symbol}' for symbol in list_of_symbols if self.check_subscribed(topic, symbol)]

    def _handle_orderbooks(self, msg):
        symbol = msg['data']['s']
        self._orderbooks_timestamps[symbol] = float(msg['ts'])
        if msg['type'] == 'snapshot':
            self._orderbooks[symbol] = {
                'bids': [[float(elem[0]), float(elem[1])] for elem in msg['data']['b']],
                'asks': [[float(elem[0]), float(elem[1])] for elem in msg['data']['a']],
            }
        else:
            for side in ['bids', 'asks']:
                _col = side[0]
                _new = False
                if len(msg['data'][_col]) > 0:
                    price_list = [elem[0]
                                  for elem in self._orderbooks[symbol][side]]
                for elem in msg['data'][_col]:
                    try:
                        idx = price_list.index(p := float(elem[0]))
                        self._orderbooks[symbol][side][idx] = [
                            p, float(elem[1])]
                    except:
                        self._orderbooks[symbol][side].append(
                            [p, float(elem[1])])
                        _new = True
                self._orderbooks[symbol][side] = [
                    elem for elem in self._orderbooks[symbol][side] if elem[1] != 0]
                if _new:
                    _reverse = True if side == 'bids' else False
                    self._orderbooks[symbol][side] = sorted(
                        self._orderbooks[symbol][side], key=lambda x: x[0], reverse=_reverse)

    def _handle_trades(self, msg, _max_len=500):
        symbol = msg['data'][0]['s']
        if symbol in self._trades.keys():
            self._trades[symbol] += msg['data']
        else:
            self._trades[symbol] = msg['data']

        self._trades[symbol] = self._trades[symbol][-1*_max_len:]

    def _handle_tickers(self, msg):
        symbol = msg['data']['symbol']
        if msg['type'] == 'snapshot':
            self._tickers[symbol] = msg['data']
        else:
            for key in msg['data'].keys():
                self._tickers[symbol][key] = msg['data'][key]

    def _handle_private_data(self, msg, _max_len=150):
        if msg['topic'] == 'position':
            self._position = msg['data'][0]
        elif msg['topic'] == 'execution':
            self._fills += msg['data']
            self._fills = self._fills[-1*_max_len:]
        elif msg['topic'] == 'order':
            for data in msg['data']:
                self._orders[data['orderId']] = data
            if len(self._orders) > _max_len:
                self._orders = {k: self._orders[k] for k in list(
                    self._orders.keys())[-1*_max_len:]}
        else:
            try:
                if msg['data'][0]['accountType'] == 'UNIFIED':
                    data = msg['data'][0]
                    data['time'] = msg['creationTime']
                    self._wallet = data
            except:
                self.logger.info(msg)

    def get_orderbook(self, symbol, depth=50):
        """
            symbols: list or str
        """
        if symbol is None:
            symbol = []
        elif type(symbol) == str:
            symbol = [symbol]

        topic = f'orderbook.{depth}'
        subscription_args = self.prepare_subscription_args(topic, symbol)

        if len(subscription_args) > 0:
            self._ws_subscribe(
                topic, self._handle_orderbooks, subscription_args)
        try:
            return self._orderbooks[symbol[0]]
        except:
            return None

    def get_trade(self, symbol):
        """
            symbols: list or str
        """
        if symbol is None:
            symbol = []
        elif type(symbol) == str:
            symbol = [symbol]

        topic = 'publicTrade'
        subscription_args = self.prepare_subscription_args(topic, symbol)

        if len(subscription_args) > 0:
            self._ws_subscribe(topic, self._handle_trades, subscription_args)

        try:
            return self._trades[symbol[0]]
        except:
            return None

    def get_ticker(self, symbol):
        """
            symbols: list or str
        """
        if symbol is None:
            symbol = []
        elif type(symbol) == str:
            symbol = [symbol]

        topic = 'tickers'
        subscription_args = self.prepare_subscription_args(topic, symbol)

        if len(subscription_args) > 0:
            self._ws_subscribe(topic, self._handle_tickers, subscription_args)

        try:
            return self._tickers[symbol[0]]
        except:
            return None

    # Private topics
    def subscribe_private_channels(self):
        for topic in self.ws.private_topics:
            if self.check_subscribed(topic, topic):
                self._ws_subscribe(topic, self._handle_private_data, [topic])

    # def position_stream(self):
    #     """
    #     https://bybit-exchange.github.io/docs/linear/#t-websocketposition
    #     """
    #     topic = "position"
    #     self._ws_subscribe(topic, self._handle_position, [topic])

    # def execution_stream(self, callback):
    #     """
    #     https://bybit-exchange.github.io/docs/linear/#t-websocketexecution
    #     """
    #     topic = "execution"
    #     self._ws_subscribe(topic=topic, callback=callback)

    # def order_stream(self, callback):
    #     """
    #     https://bybit-exchange.github.io/docs/linear/#t-websocketorder
    #     """
    #     topic = "order"
    #     self._ws_subscribe(topic=topic, callback=callback)

    # def stop_order_stream(self, callback):
    #     """
    #     https://bybit-exchange.github.io/docs/linear/#t-websocketstoporder
    #     """
    #     topic = "stop_order"
    #     self._ws_subscribe(topic=topic, callback=callback)

    # def wallet_stream(self, callback):
    #     """
    #     https://bybit-exchange.github.io/docs/linear/#t-websocketwallet
    #     """
    #     topic = "wallet"
    #     self._ws_subscribe(topic, self._handle_, [topic])
