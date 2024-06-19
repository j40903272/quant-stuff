from pluto.exchange import BinanceExchange, BingxExchange
from pluto.websocket import BinanceFutureBooktickerSocket, OKXOrderbookWebsocket, BingxOrderbookWebsocket
import threading
import configparser

import pandas as pd
import time
from datetime import datetime

class OrderbookSocketManager:
    def __init__(self, symbol, exchange):
        self.table = {}
        self.symbol = symbol
        self.socket_instance = None
        self.socket_thread = None
        self.exchange = exchange
        self.startTime = time.time()
    
    def update_best_bid_ask(self, data):

        time = datetime.now()

        if(time.strftime('%M') == '10' and time.strftime('%S') == '00'):
            self.table = {}

        if((time.strftime('%M')[1] == '4' or time.strftime('%M')[1] == '9' ) and time.strftime('%S') == '00'):
            df = pd.DataFrame.from_dict(self.table, orient="index")
            temp = self.symbol.replace('-','')
            df.to_csv(f'symbols_Raw/{self.exchange}/{temp}.csv', index=False)
        try:
            best_bid = data['bid']
            best_ask = data['ask']

            if best_bid and best_ask:
                self.table[data['t']] = {
                    'timestamp': data['t'],
                    'bid': best_bid[0],
                    #'bid_qty': best_bid[1],
                    'ask': best_ask[0],
                    #'ask_qty': best_ask[1]
                }

        except Exception as e:
            pass

    def start_socket(self):
        pass

    def stop_socket(self):
        if self.socket_instance:
            self.socket_instance.ws.close()

class BinanceOrderbookSocketManager(OrderbookSocketManager):
    def __init__(self, symbol, exchange = 'binance'):
        super().__init__(symbol, exchange)

    def start_socket(self):
        self.socket_instance = BinanceFutureBooktickerSocket(self.symbol)
        self.socket_instance.handler = lambda data: self.update_best_bid_ask(data)
        self.socket_thread = threading.Thread(target=self.socket_instance.start)
        self.socket_thread.start()

class OkxOrderbookSocketManager(OrderbookSocketManager):
    def __init__(self, symbol, exchange = 'okx'):
        super().__init__(symbol, exchange)

    def start_socket(self):
        self.socket_instance = OKXOrderbookWebsocket(self.symbol)
        self.socket_instance.handler = lambda data: self.update_best_bid_ask(data)
        self.socket_thread = threading.Thread(target=self.socket_instance.start)
        self.socket_thread.start()

#symbols = ['LSK-USDT', 'ENS-USDT', 'THETA-USDT']

# bingx_orderbook_socket = None
# bingx_orderbook_socket_thread = None

# bingx_best_bid = 0
# bingx_best_ask = 0
# def update_bingx_best_bid_ask(data):
#     try:
#         bingx_best_bid = float(data['bids'][0][0])
#         bingx_best_ask = float(data['asks'][0][0])
#         if bingx_best_bid and bingx_best_ask:
#             bingx[data['t']] = {
#                 'timestamp': data['t'],
#                 'bid': bingx_best_bid,
#                 'ask': bingx_best_ask,
#             }
#     except Exception as e:
#         pass

try:
    config = configparser.ConfigParser()
    config.read('./config.ini')

    # 如果你的檔案只有一個section，你可以使用config['SectionName']
    # 這裡假設section名稱為'Symbols'
    symbols = config['DEFUALT']['symbols'].split(',')

except FileNotFoundError:
    print(f"找不到檔案")
except Exception as e:
    print(f"讀取檔案時發生錯誤：{str(e)}")

for symbol in symbols:
    symbol = symbol + '-USDT'
    binance_orderbook_manager = BinanceOrderbookSocketManager(symbol.replace('-',''))
    binance_orderbook_manager.start_socket()
    
    okx_orderbook_manager = OkxOrderbookSocketManager(symbol)
    okx_orderbook_manager.start_socket()



# Bingx
# bingx_orderbook_socket = BingxOrderbookWebsocket(symbol)
# bingx_orderbook_socket.handler = lambda data: update_bingx_best_bid_ask(data)
# bingx_orderbook_socket_thread = threading.Thread(target=bingx_orderbook_socket.start, kwargs={'auto_reconnect': False})
# bingx_orderbook_socket_thread.start()



while(1):
    try:
        pass
        # df = pd.DataFrame.from_dict(binance, orient="index")
        # df.to_csv(f'symbols_Raw/binance/{symbol}', index=True)

        # df = pd.DataFrame.from_dict(bingx, orient="index")
        # df.to_csv(f'symbols_Raw/okx/{symbol}', index=False)
        # df = pd.DataFrame(bingx)
        # df.to_csv(f'symbols_Raw/okx/{symbol}', index=True)
    except Exception as e:
        pass
    pass

