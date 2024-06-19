from pluto.websocket import BinanceFutureBooktickerSocket, BingxOrderbookWebsocket
import threading
import requests
import numpy as np
import time

MAX_LENGTH = 8999

def calculate_correlation(list1, list2):
    if len(list1) == 0 or len(list2) == 0:
        return np.nan  # Return None if one of the lists is empty

    if len(list1) != len(list2):
        return np.nan  # Return None if lists are not of equal length

    correlation_matrix = np.corrcoef(list1, list2)
    correlation = correlation_matrix[0, 1]  # Extract the correlation coefficient
    return correlation

def fetch_common_usdt_symbols(rank = 50):
    # Fetch symbols from BingX
    bingx_response = requests.get('https://open-api.bingx.com/openApi/swap/v2/quote/contracts')
    if bingx_response.status_code == 200:
        bingx_symbols = [symbol['symbol'].replace("-", "") for symbol in bingx_response.json()['data'] if symbol['symbol'].endswith('USDT')]
    else:
        raise Exception(f"BingX Error: {bingx_response.status_code}\n{bingx_response.text}")

    # Fetch symbols from Binance
    binance_response = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr")
    if binance_response.status_code == 200:
        data = binance_response.json()
        # Sort by 24-hour trading volume * weighted average price, descending
        sorted_data = sorted(data, key=lambda x: float(x['volume']) * float(x['weightedAvgPrice']), reverse=True)
        #print(sorted_data)
        # Extract symbols for the top 150 futures, assuming the sorting has already been done
        binance_symbols = [item['symbol'] for item in sorted_data if 'USDT' in item['symbol']]
    else:
        raise Exception(f"Binance Error: {binance_response.status_code}\n{binance_response.text}")
    return list(set(binance_symbols[:50]) & set(bingx_symbols))

class MarketMonitor:
    def __init__(self, trading_symbols):
        # Symbols being monitored
        self.traded_symbols = trading_symbols
        # Websocket connections for order book data
        self.binance_orderbook_socket = None
        self.bingx_orderbook_socket = None
        self.binance_orderbook_thread = None
        self.bingx_orderbook_thread = None
        self.market_data_thread = None

        # Best bid and ask prices
        self.ticker = {
            'binance_ask':{},
            'binance_bid':{},
            'bingx_ask':{},
            'bingx_bid':{},
        }

        # Historical data
        self.history = {
            'binance_ask':{},
            'binance_bid':{},
            'bingx_ask':{},
            'bingx_bid':{},
        }

        # Statistical data
        self.statistics = {
            'midprice_correlation':{},
            'volatility':{}
        }

        self.initialize_symbol_data(trading_symbols)
        self.initialize_sockets()

    def update_binance_best_bid_ask(self, order_data):
        #print('binance:', order_data)
        try:
            self.ticker['binance_bid'][order_data['symbol']] = float(order_data['bids'][0][0])
            self.ticker['binance_ask'][order_data['symbol']] = float(order_data['asks'][0][0])
        except Exception as e:
            # Log error or handle exception
            pass

    def update_bingx_best_bid_ask(self, order_data):
        #print('bingx:', order_data)
        try:
            self.ticker['bingx_bid'][order_data['symbol']] = float(order_data['bids'][0][0])
            self.ticker['bingx_ask'][order_data['symbol']] = float(order_data['asks'][-1][0])
        except Exception as e:
            # Log error or handle exception
            pass

    def initialize_symbol_data(self, symbol):
        for symbol in self.traded_symbols:
            for key in self.ticker.keys():
                self.ticker[key][symbol] = []

            for key in self.history.keys():
                self.history[key][symbol] = []

            for key in self.statistics.keys():
                self.statistics[key][symbol] = []

    def update_market_data(self):
        while 1:
            for symbol in self.traded_symbols:
                if self.ticker['binance_bid'][symbol] and self.ticker['binance_ask'][symbol] \
                        and self.ticker['bingx_bid'][symbol] and self.ticker['bingx_ask'][symbol]:
                    self.update_historical_data(symbol)
                    self.update_statistics(symbol)
            time.sleep(0.1)

    def update_historical_data(self, symbol):
        for key in self.ticker.keys():
            if len(self.history[key][symbol]) > MAX_LENGTH:
                self.history[key][symbol] = self.history[key][symbol][1:MAX_LENGTH+1]
            self.history[key][symbol].append(self.ticker[key][symbol])

    def update_statistics(self, symbol):
        # Binance
        bids_binance, asks_binance = np.array(self.history['binance_bid'][symbol]), np.array(self.history['binance_ask'][symbol])
        midprices_binance = (bids_binance + asks_binance) / 2.0
        spread_binance = (asks_binance - bids_binance) / midprices_binance * 100
        spread_SD_binance = np.std(spread_binance)
        midprice_SD_binance = np.std(midprices_binance) / midprices_binance[-1]



        # BingX
        bids_bingx, asks_bingx = np.array(self.history['bingx_bid'][symbol]), np.array(self.history['bingx_ask'][symbol])
        midprices_bingx = (bids_bingx + asks_bingx) / 2.0
        spread_bingx = (asks_bingx - bids_bingx) / midprices_bingx * 100
        spread_SD_bingx = np.std(spread_bingx)
        midprice_SD_bingx = np.std(midprices_bingx) / midprices_bingx[-1]


        # Diff
        midprice_correlation = calculate_correlation(midprices_binance, midprices_bingx)
        if not np.isnan(midprice_correlation):
            if len(self.statistics['midprice_correlation'][symbol]) > MAX_LENGTH:
                self.statistics['midprice_correlation'][symbol] = self.statistics['midprice_correlation'][symbol][1:MAX_LENGTH+1]
            self.statistics['midprice_correlation'][symbol].append(midprice_correlation)

        # Update return
        self.statistics['volatility'][symbol] = {
            'Binance': {
                'spread': np.mean(spread_binance),
                'spread_SD': abs(spread_SD_binance),
                'midprice_SD':abs(midprice_SD_binance)
            },
            'Bingx': {
                'spread': np.mean(spread_bingx),
                'spread_SD': abs(spread_SD_bingx),
                'midprice_SD':abs(midprice_SD_bingx)
            }
        }

    def initialize_sockets(self):
        # Initialize Binance order book websocket
        self.binance_orderbook_socket = BinanceFutureBooktickerSocket(self.traded_symbols)
        self.binance_orderbook_socket.handler = lambda data: self.update_binance_best_bid_ask(data)
        self.binance_orderbook_thread = threading.Thread(target=self.binance_orderbook_socket.start, kwargs={'auto_reconnect': False})
        self.binance_orderbook_thread.start()

        # Initialize Bingx order book websocket
        self.bingx_orderbook_socket = BingxOrderbookWebsocket(self.traded_symbols)
        self.bingx_orderbook_socket.handler = lambda data: self.update_bingx_best_bid_ask(data)
        self.bingx_orderbook_thread = threading.Thread(target=self.bingx_orderbook_socket.start, kwargs={'auto_reconnect': False})
        self.bingx_orderbook_thread.start()

        self.market_data_thread = threading.Thread(target=self.update_market_data)
        self.market_data_thread.start()