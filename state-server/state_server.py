import os
import json
import configparser
import threading
import time
from format import Formatter

from pluto.exchange import BinanceExchange, BingxExchange
import asyncio
from pluto.websocket import BinanceOrderbookSocket, BingxOrderbookWebsocket, BinanceUserSocket

class StateServer:

    def __init__(self):

        with open(os.path.join("./", "configs/TRBUSDT.json")) as f:
            config = json.load(f)

        api_key_ini = configparser.ConfigParser()
        api_key_ini.read(filenames=config["api_key_path"])

        self.binance = BinanceExchange(api_key_ini["binance"]['key'],api_key_ini["binance"]['secret'])
        self.bingx = BingxExchange(api_key_ini["bingx"]['key'],api_key_ini["bingx"]['secret'])
        self.formatter = Formatter()
        self.currnet_orders = []

        self.symbol = "XRP-USDT"

        # Bot state
        self.state = "init"
        self.bingx_best_bid = None
        self.bingx_best_ask = None
        self.binance_best_bid = None
        self.binance_best_ask = None
        self.binance_now_order = None
        self.binance_now_position = None
        self.binance_filled_qty = None

        self.bingx_orderbook = None
        self.binance_orderbook = None

        # # Binance User Socket
        # binance_user_socket = BinanceUserSocket(api_key_ini["binance"]['key'],api_key_ini["binance"]['secret'])
        # binance_user_socket.handler = lambda data: self.update_binance_user_status(data)
        # threading.Thread(target=binance_user_socket.start, kwargs={'auto_reconnect': False}).start()

        # # Binance
        # self.binance_socket = BinanceOrderbookSocket(self.symbol)
        # self.binance_socket.handler = lambda data: self.update_binance_best_bid_ask(data)
        # threading.Thread(target=self.binance_socket.start, kwargs={'auto_reconnect': False}).start()

        # # Bingx
        # self.bingx_socket = BingxOrderbookWebsocket(self.symbol)
        # self.bingx_socket.handler = lambda data: self.update_bingx_best_bid_ask(data)
        # threading.Thread(target=self.bingx_socket.start, kwargs={'auto_reconnect': False}).start()

    def get_open_positions(self, exchange, symbol = None):
        if(exchange == 'binance'):
            binance = asyncio.run(self.binance.fetch_position(symbol))
            binance['data'] = [data for data in binance.get('data', []) if abs(float(data['positionAmt'])) > 0 ]
            print(binance['data'])
            return {"message": binance['message'], "data":self.formatter.to_binanace_position(binance['data'])}
        elif(exchange == 'bingx'):
            bingx = asyncio.run(self.bingx.fetch_position(self.formatter.to_bingx_symbol(symbol)))
            return {"message": bingx['message'], "data":self.formatter.to_bingx_position(bingx['data'])}

    def get_pending_orders(self, exchange, symbol = None):
        if(exchange == 'binance'):
            binance = asyncio.run(self.binance.fetch_pending_orders(symbol))
            print(binance)
            return {"message": binance['message'], "data": self.formatter.to_binanace_order(binance['data'])}
        elif(exchange == 'bingx'):
            bingx = asyncio.run(self.bingx.fetch_pending_orders(self.formatter.to_bingx_symbol(symbol)))
            return {"message": bingx['message'], "data": self.formatter.to_bingx_order(bingx['data'])}

    def get_best_offer(self, exchange, symbol):
        result = {
            'exchange':exchange,
            'symbol':symbol,
            'bids':self.binance_best_bid,
            'asks':self.binance_best_ask,
        }
        return result
    
    def get_symbols(self, exchange):
        if(exchange == 'binance'):
            binance = asyncio.run(self.binance.fetch_symbols())
            return { "message": binance['message'], "data": binance['data']}
        elif(exchange == 'bingx'):
            bingx = asyncio.run(self.bingx.fetch_symbols())
            return {"message": bingx['message'], "data": bingx['data']}
    
    def get_balance(self, exchange):
        if(exchange == 'binance'):
            binance = asyncio.run(self.binance.fetch_balance())
            return { "message": binance['message'], "data": binance['data']}
        elif(exchange == 'bingx'):
            bingx = asyncio.run(self.bingx.fetch_balance())
            return {"message": bingx['message'], "data": bingx['data']}

    def get_markPrice(self, exchange, symbol):
        if(exchange == 'binance'):
            binance = asyncio.run(self.binance.fetch_markPrice(symbol))
            return {"message": binance['message'], "data": binance['data']}
        elif(exchange == 'bingx'):
            bingx = asyncio.run(self.bingx.fetch_markPrice(self.formatter.to_bingx_symbol(symbol)))
            return {"message": bingx['message'], "data": bingx['data']}
    def get_fundingRate(self, exchange, symbol):
        if(exchange == 'binance'):
            binance = asyncio.run(self.binance.fetch_fundingRate(symbol))
            return {"message": binance['message'], "data": binance['data']}
        elif(exchange == 'bingx'):
            bingx = asyncio.run(self.bingx.fetch_fundingRate(self.formatter.to_bingx_symbol(symbol)))
            return {"message": bingx['message'], "data": bingx['data']}
    
    def get_historical_orders(self, exchange, symbol, startTime = int(time.time() * 1000 - (90 * 24 * 60 * 59 * 1000)), endTime = int(time.time() * 1000)):
        if(exchange == 'binance'):
            binance = asyncio.run(self.binance.fetch_historical_orders(symbol, startTime, endTime))
            return {"message": binance['message'], "data": self.formatter.to_binance_historical_orders(binance['data'])}
        elif(exchange == 'bingx'):
            bingx = asyncio.run(self.bingx.fetch_historical_orders(self.formatter.to_bingx_symbol(symbol), startTime, endTime))
            return {"message": bingx['message'], "data": self.formatter.to_bingx_historical_orders(bingx['data'])}

    def cancel_order(self, exchange, symbol, orderId):
        if(exchange == 'binance'):
            binance = asyncio.run(self.binance.cancel_order(symbol, orderId))
            return {"message": binance['message']}
        elif(exchange == 'bingx'):
            bingx = asyncio.run(self.bingx.cancel_order(self.formatter.to_bingx_symbol(symbol), orderId))
            return {"message": bingx['message']}

    def place_order(self, exchange, symbol, type, side, positionSide, quantity, price = None, timeforce = "GTC"):
        if(exchange == 'binance'):
            binance = asyncio.run(self.binance.place_order(symbol, type, side, positionSide, quantity, price, timeforce))
            return {"message": binance['message']}
        elif(exchange == 'bingx'):
            bingx = asyncio.run(self.bingx.place_order(self.formatter.to_bingx_symbol(symbol), type, side, positionSide, quantity, price, timeforce))
            return {"message": bingx['message']}


    def update_binance_best_bid_ask(self, data):
        try:
            self.binance_best_bid = float(data['bids'][0][0])   
            self.binance_best_ask = float(data['asks'][0][0])
        except Exception as e:
            pass

    def update_bingx_best_bid_ask(self, data):
        try:
            self.bingx_best_bid = float(data['bids'][0][0])
            self.bingx_best_ask = float(data['asks'][-1][0])
        except Exception as e:
            pass

    def update_binance_user_status(self, data):
        if data['e'] != 'ORDER_TRADE_UPDATE':
            return
        if data['o']['s'] != self.binance_symbol:
            return
        if data['o']['X'] != 'FILLED':
            return
        if data['o']['i'] != self.binance_now_order['orderId']:
            return
        self.binance_now_order = None
        self.binance_filled_qty = float(data['o']['q'])
    
    def update_symbols(self, exchange, symbols):
        if symbols:
            if(exchange == 'binance'):
                 # Binance
                self.binance_socket.ws.close()
                self.binance_socket = BinanceOrderbookSocket(symbols)
                self.binance_socket.handler = lambda data: self.update_binance_best_bid_ask(data)
                threading.Thread(target=self.binance_socket.start, kwargs={'auto_reconnect': False}).start()
                return "success"
            elif(exchange == 'bingx'):
                self.bingx_socket.ws.close()
                self.bingx_socket = BingxOrderbookWebsocket(symbols )
                self.bingx_socket.handler = lambda data: self.update_bingx_best_bid_ask(data)
                threading.Thread(target=self.bingx_socket.start, kwargs={'auto_reconnect': False}).start()
                return "success"
        return 'error'
    