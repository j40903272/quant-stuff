# %%

from unicorn_binance_websocket_api.manager import BinanceWebSocketApiManager
from .process_streams import UnicornFy
import time
import threading
import re
from typing import DefaultDict, List, Dict
from collections import defaultdict


class BinanceWebsocketClient():
    def __init__(self, api_key='', api_secret='', subaccount_name='', exchange="binance.com") -> None:
        self.exchange = exchange
        self._api_key = api_key  # TODO: Place your API key here
        self._api_secret = api_secret  # TODO: Place your API secret here
        self._subaccount = subaccount_name
        self.ws = BinanceWebSocketApiManager(
            exchange=exchange, throw_exception_if_unrepairable=True)
        self.stream_id = ''
        self._reset_data()

    def _reset_data(self) -> None:
        print('*'*10, 'reset data', '*'*10)
        self._subscriptions: List[Dict] = []
        self._aggtrades: DefaultDict[str, Dict] = defaultdict(list)
        self._trades: DefaultDict[str, Dict] = defaultdict(list)
        self._klines: DefaultDict[str, Dict] = defaultdict(list)
        # self._minitickers: DefaultDict[str, Dict] = defaultdict(dict)
        # self._tickers: DefaultDict[str, Dict] = defaultdict(dict)
        self._orderbook_update_id: DefaultDict[str, str] = defaultdict(dict)
        self._orderbooks: DefaultDict[str, Dict[str, DefaultDict[float, float]]] = defaultdict(
            lambda: {side: defaultdict(float) for side in {'bids', 'asks'}})
        self._mark_prices = {}

    def _reset_orderbook(self, market: str) -> None:
        if market in self._orderbooks:
            del self._orderbooks[market]
        if market in self._orderbook_update_id:
            del self._orderbook_update_id[market]

    def _unified(self, text):
        if re.match(r'[a-zA-Z0-9]{41,43}', text) is None:
            return str(text).upper()
        else:
            return str(text)

    def _process_stream_data(self, received_stream_data_json, stream_buffer_name="False"):
        if self.exchange == "binance.com" or self.exchange == "binance.com-testnet":
            return UnicornFy.binance_com_websocket(received_stream_data_json)
        elif self.exchange == "binance.com-futures" or self.exchange == "binance.com-futures-testnet":
            return UnicornFy.binance_com_futures_websocket(received_stream_data_json)
        elif self.exchange == "binance.com-margin" or self.exchange == "binance.com-margin-testnet":
            return UnicornFy.binance_com_margin_websocket(received_stream_data_json)
        elif self.exchange == "binance.com-isolated_margin" or self.exchange == "binance.com-isolated_margin-testnet":
            return UnicornFy.binance_com_margin_websocket(received_stream_data_json)
        elif self.exchange == "binance.je":
            return UnicornFy.binance_je_websocket(received_stream_data_json)
        elif self.exchange == "binance.us":
            return UnicornFy.binance_us_websocket(received_stream_data_json)
        else:
            print("Not a valid exchange: " + str(self.exchange))

    def _handle_aggtrade(self, message, _max_len=500):
        market = self._unified(message['symbol'])
        self._aggtrades[market].append(message)
        self._aggtrades[market] = self._aggtrades[market][-1*_max_len:]

    def _handle_trade(self, message, _max_len=500):
        market = self._unified(message['symbol'])
        self._trades[market].append(message)
        self._trades[market] = self._trades[market][-1*_max_len:]

    def _handle_kline(self, message, _max_len=1000):
        market = self._unified(message['symbol'])
        self._klines[market] = message
        self._klines[market] = self._klines[market][-1*_max_len:]

    def _handle_ticker(self, message):
        pass
        # self._tickers[message['symbol']].append(message)

    def _handle_miniticker(self, message):
        pass
        # self._minitickers[message['symbol']].append(message)

    def _handle_depth(self, message):
        market = self._unified(message['symbol'])
        self._orderbook_update_id[market] = message['last_update_id']
        self._orderbooks[market] = {'bids': [[float(o[0]), float(
            o[1])]for o in message['bids']], 'asks': [[float(o[0]), float(o[1])]for o in message['asks']]}
    
    def _handle_mark_price(self, message):
        market = self._unified(message['symbol'])
        self._mark_prices[market] = message

    def _handle_anything_else(self, message):
        pass

    def _handler(self):
        while True:
            if self.ws.is_manager_stopping():
                return None
            _stream_data = self.ws.pop_stream_data_from_stream_buffer()
            if _stream_data is False:
                time.sleep(0.001)
            else:
                _unified_stream_data = self._process_stream_data(_stream_data)
                try:
                    if _unified_stream_data['event_type'] == "aggTrade":
                        self._handle_aggtrade(_unified_stream_data)
                    elif _unified_stream_data['event_type'] == "trade":
                        self._handle_trade(_unified_stream_data)
                    elif _unified_stream_data['event_type'] == "kline":
                        self._handle_kline(_unified_stream_data)
                    elif _unified_stream_data['event_type'] == "24hrMiniTicker":
                        self._handle_miniticker(_unified_stream_data)
                    elif _unified_stream_data['event_type'] == "24hrTicker":
                        self._handle_ticker(_unified_stream_data)
                    elif _unified_stream_data['event_type'] == "depth":
                        self._handle_depth(_unified_stream_data)
                    elif "markPrice" in _unified_stream_data['event_type']:
                        self._handle_mark_price(_unified_stream_data)
                    else:
                        self._handle_anything_else(_unified_stream_data)
                except KeyError:
                    self._handle_anything_else(_unified_stream_data)
                except TypeError:
                    pass

    def _subscribe(self, subscription):
        """
            subscription = {'channels': '', 'markets': ''}
        """
        if subscription not in self._subscriptions:
            if self.stream_id == '':
                self.stream_id = self.ws.create_stream(**subscription)
                # start a worker process to move the received stream_data from the stream_buffer to a print function
                self.worker_thread = threading.Thread(
                    target=self._handler, args=())
                self.worker_thread.start()
            else:
                self.ws.subscribe_to_stream(self.stream_id, **subscription)
            self._subscriptions.append(subscription)

    def _unsubscribe(self, subscription):
        self.ws.subscribe_to_stream(self.stream_id, **subscription)
        while subscription in self._subscriptions:
            self._subscriptions.remove(subscription)

    def _close_streams(self):
        self.ws.stop_manager_with_all_streams()
        self.stream_id = ''

    def get_orderbook(self, market, depth = 10):
        market = self._unified(market)
        subscription = {'channels': f'depth{depth}', 'markets': market}
        if(subscription not in self._subscriptions):
            self._subscribe(subscription)
        if len(self._orderbooks[market]['bids']) == 0 or len(self._orderbooks[market]['asks']) == 0:
            return None
        else:
            return self._orderbooks[market]
    
    def get_mark_price(self, market):
        market = self._unified(market)
        subscription = {'channels': 'markPrice', 'markets': market}
        if(subscription not in self._subscriptions):
            self._subscribe(subscription)
        try:
            return self._mark_prices[market]
        except:
            return None

    def get_orderbook_update_id(self, market):
        market = self._unified(market)
        return self._orderbook_update_id[market]

    def get_trades(self, market):
        market = self._unified(market)
        subscription = {'channels': 'trade', 'markets': market}
        if(subscription not in self._subscriptions):
            self._subscribe(subscription)
        try:
            return self._trades[market]
        except:
            return None
        

    def get_aggtrades(self, market):
        market = self._unified(market)
        return self._aggtrades[market]

    def monitor(self):
        self.ws.print_summary()

# %%
