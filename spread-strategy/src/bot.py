import os
import logging
def setup_logger(module_name=__name__):
    fmt = "[%(asctime)s][%(levelname)s][{}.%(funcName)s] %(message)s".format(module_name)
    path = module_name + ".log.txt"
    logging.basicConfig(format=fmt, level=logging.INFO if os.environ.get('DEBUG') is None else logging.DEBUG)
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger("urllib3").propagate = False
    logging.getLogger('asyncio').propagate = False
    return logging.getLogger(__name__)

import json
import time
import asyncio
import threading

from pluto.exchange import BinanceExchange, BingxExchange
from pluto.websocket import BinanceFutureBooktickerSocket, BingxOrderbookWebsocket, BinanceUserSocket


class Bot():
    def __init__(self, config):
        # Strategy config
        self.symbol = config['symbol']
        self.binance_symbol = self.symbol.replace("-", "")
        self.precision = config['precision']
        self.bingx_symbol = self.symbol
        self.leverage = config['leverage']
        self.want_to_earn = config['want_to_earn']
        self.close_tolerant = config['close_tolerant']
        self.default_qty = config['default_qty']
        self.want_to_earn = config['want_to_earn']
        self.binance_maker_fee = config['binance_maker_fee']
        self.binance_taker_fee = config['binance_taker_fee']
        self.bingx_maker_fee = config['bingx_maker_fee']
        self.bingx_taker_fee = config['bingx_taker_fee']
        self.renew_interval_ms = config['renew_interval_ms']
        self.binance_apikey = config['binance_apikey']
        self.binance_secret = config['binance_secret']
        self.bingx_apikey = config['bingx_apikey']
        self.bingx_secret = config['bingx_secret']

        self.logger = setup_logger(self.symbol)

        self.binance_exchange = BinanceExchange(self.binance_apikey, self.binance_secret)
        self.bingx_exchange = BingxExchange(self.bingx_apikey, self.bingx_secret)

        # Bot state
        self.state = "init"
        self.bingx_best_bid = None
        self.bingx_best_ask = None
        self.binance_best_bid = None
        self.binance_best_ask = None
        self.binance_mutax = threading.Lock()
        self.binance_now_order = None
        self.binance_now_position = None
        self.binance_filled_qty = None
        self.binance_cancel_order_cooldown_ms = 1000
        self.binance_cancel_order_cooldown = 0

        # Bot socket
        self.binance_user_socket = None
        self.binance_orderbook_socket = None
        self.bingx_orderbook_socket = None
        self.binance_user_socket_thread = None
        self.binance_orderbook_socket_thread = None
        self.bingx_orderbook_socket_thread = None

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
        self.binance_mutax.acquire()
        if self.binance_now_order != None and data['e'] == 'ORDER_TRADE_UPDATE' and data['o']['s'] == self.binance_symbol:
            if data['o']['X'] == 'CANCELED' and data['o']['i'] == self.binance_now_order['orderId']:
                self.logger.debug(f"(ws): Order {data['o']['i']} canceled")
                self.binance_now_order = None
            elif data['o']['X'] == 'FILLED' and data['o']['i'] == self.binance_now_order['orderId']:
                self.logger.info(f"(ws): Order {data['o']['i']} filled")
                self.binance_now_order = None
                self.binance_filled_qty = float(data['o']['q'])
                self.logger.debug(f"BingX 期望 take {self.bingx_best_bid} {self.bingx_best_ask}")
            elif data['o']['X'] == 'NEW':
                self.logger.debug(f"(ws): Order {data['o']['i']} new")
                pass
            else:
                self.logger.debug(f"Unknown order status: " + json.dumps(data))
        self.binance_mutax.release()

    def init(self):
        # 檢查前次狀態
        orders = asyncio.run(self.binance_exchange.fetch_open_orders(self.binance_symbol))
        if len(orders) > 0:
            self.logger.info("有未成交單，系統自動取消")
            for order in orders:
                self.logger.info(f"取消未成交單 {order['orderId']}")
                asyncio.run(self.binance_exchange.cancel_order(self.binance_symbol, order['orderId']))

        # set Binance leveragem, 1x
        res = asyncio.run(self.binance_exchange.set_leverage(self.binance_symbol, self.leverage))
        if res.get('leverage', None) != self.leverage:
            raise Exception("set Binance leverage failed")
        self.logger.info(f"set Binance leverage {self.leverage}x")

        # set Bingx leverage, 1x
        res = asyncio.run(self.bingx_exchange.set_leverage(self.bingx_symbol, self.leverage))
        if res['data'].get('leverage', None) != self.leverage:
            raise Exception("set Bingx leverage failed")
        self.logger.info(f"set Bingx leverage {self.leverage}x")

        # Binance User Socket
        self.binance_user_socket = BinanceUserSocket(self.binance_apikey, self.binance_secret)
        self.binance_user_socket.handler = lambda data: self.update_binance_user_status(data)
        self.binance_user_socket_thread = threading.Thread(target=self.binance_user_socket.start, kwargs={'auto_reconnect': False})
        self.binance_user_socket_thread.start()

        # Start websocket for updating best bid and ask

        # Binance
        self.binance_orderbook_socket = BinanceFutureBooktickerSocket(self.symbol)
        self.binance_orderbook_socket.handler = lambda data: self.update_binance_best_bid_ask(data)
        self.binance_orderbook_socket_thread = threading.Thread(target=self.binance_orderbook_socket.start, kwargs={'auto_reconnect': False})
        self.binance_orderbook_socket_thread.start()

        # Bingx
        self.bingx_orderbook_socket = BingxOrderbookWebsocket(self.symbol)
        self.bingx_orderbook_socket.handler = lambda data: self.update_bingx_best_bid_ask(data)
        self.bingx_orderbook_socket_thread = threading.Thread(target=self.bingx_orderbook_socket.start, kwargs={'auto_reconnect': False})
        self.bingx_orderbook_socket_thread.start()

        # FIXME: 這邊要等待 binance_user_socket 連線成功，現在先 workaround
        time.sleep(1)

        # wait best bid and ask ready
        while True:
            if self.binance_best_bid and self.binance_best_ask and self.bingx_best_bid and self.bingx_best_ask:
                break

        self.logger.info("Bot initialized")

        binance_position = asyncio.run(self.binance_exchange.fetch_position(self.binance_symbol))
        binance_has_position = False
        for position in binance_position:
            if position.get('positionAmt', 0) != 0 and float(position.get('positionAmt', 0)) == self.default_qty:
                binance_has_position = True
                break
        bingx_position = asyncio.run(self.bingx_exchange.fetch_position(self.bingx_symbol))
        bingx_has_position = False
        for position in bingx_position:
            if position.get('positionAmt', 0) != 0 and float(position.get('positionAmt', 0)) == self.default_qty:
                bingx_has_position = True
                break

        if binance_has_position and bingx_has_position:
            self.logger.info("上次斷線是在 wait_close，繼續 wait_close")
            self.state = "wait_close"
            self.binance_now_position = self.default_qty
        else:
            self.state = "wait_buy_in"

    def wait_buy_in(self):
        self.binance_mutax.acquire()
        threshold = self.binance_maker_fee + self.bingx_taker_fee + self.want_to_earn
        # 掛單成交 => BingX 對衝
        if self.binance_filled_qty != None:
            self.binance_now_position = self.binance_filled_qty
            self.binance_filled_qty = None
            self.logger.debug(f"BingX 期望 take {self.bingx_best_bid} {self.bingx_best_ask}")
            res = asyncio.run(self.bingx_exchange.place_order(self.bingx_symbol, "MARKET", "SHORT", "SELL", 0, self.binance_now_position))
            self.logger.info("BingX take market order done, order id: " + str(res['orderId']))
            self.state = 'wait_close'
        # 沒成交，沒掛單 => 掛單
        elif self.binance_now_order == None:
            prefer_long_entry = round(self.bingx_best_bid * (1 - threshold), self.precision)
            prefer_long_qty = self.default_qty

            if prefer_long_entry < self.binance_best_ask:
                self.logger.info(f"place limit order {prefer_long_qty}@{prefer_long_entry}, binance: {self.binance_best_bid} {self.binance_best_ask}, bingx: {self.bingx_best_bid} {self.bingx_best_ask}")
                order = asyncio.run(self.binance_exchange.place_order(self.binance_symbol, "LIMIT", "LONG", "BUY", prefer_long_entry, prefer_long_qty))
                if order is not None and order.get('status', None) == "NEW" and order.get('orderId', None) is not None:
                    self.binance_now_order = order
                else:
                    print("Order: ", order)
                    raise Exception("hunting")
        # 沒成交，有掛單，超時 => 取消掛單
        elif self.binance_cancel_order_cooldown < time.time_ns() // 1000000:
            # 近到連手續費都不夠，就抽掉
            fee = self.binance_maker_fee + self.bingx_taker_fee
            min_limit_cond = self.bingx_best_bid <= float(self.binance_now_order['price']) / (1-fee)
            # 遠到不太可能成交，就抽掉
            max_limit_cond = self.bingx_best_bid >= float(self.binance_now_order['price']) / (1-threshold*2)
            if min_limit_cond or max_limit_cond:
                self.logger.info(f"    ==> cancel order {self.binance_now_order['orderId']}")
                res = asyncio.run(self.binance_exchange.cancel_order(self.binance_symbol, self.binance_now_order['orderId']))
                self.binance_cancel_order_cooldown = int(time.time_ns() // 1000000) + self.binance_cancel_order_cooldown_ms
                if res.get('code', None) != None:
                    if res.get('code') == -2011: # failed: {"code": -2011, "msg": "Unknown order sent."}
                        self.binance_now_order = None
                    else:
                        self.logger.error(f"    => cancel order failed: {json.dumps(res)}")
        # 沒成交，有掛單，未超時 => 等待
        else:
            pass
        self.binance_mutax.release()

    def wait_close(self):
        self.binance_mutax.acquire()
        threshold = self.binance_maker_fee + self.bingx_taker_fee + self.want_to_earn
        # 掛單成交 => BingX 平倉
        if self.binance_filled_qty != None:
            self.logger.debug(f"BingX 期望 take {self.bingx_best_bid} {self.bingx_best_ask}")
            res = asyncio.run(self.bingx_exchange.place_order(self.bingx_symbol, "MARKET", "SHORT", "BUY", 0, self.binance_filled_qty))
            self.logger.info("BingX take market order done, order id: " + str(res['orderId']))
            self.binance_now_position = None
            self.binance_filled_qty = None
            self.state = 'wait_buy_in'
        # 沒成交，沒掛單 => 掛單
        elif self.binance_now_order == None:
            prefer_short_entry = round(self.bingx_best_ask * (1 + threshold), self.precision)
            prefer_short_qty = self.binance_now_position

            if prefer_short_entry > self.binance_best_bid:
                self.logger.info(f"place limit order {prefer_short_qty}@{prefer_short_entry}, binance: {self.binance_best_bid} {self.binance_best_ask}, bingx: {self.bingx_best_bid} {self.bingx_best_ask}")
                order = asyncio.run(self.binance_exchange.place_order(self.binance_symbol, "LIMIT", "LONG", "SELL", prefer_short_entry, prefer_short_qty))
                if order is not None and order.get('status', None) == "NEW" and order.get('orderId', None) is not None:
                    self.binance_now_order = order
                else:
                    print("Order2: ", order)
                    raise Exception("hunting2")
        # 沒成交，有掛單，超時 => 取消掛單
        elif self.binance_cancel_order_cooldown < time.time_ns() // 1000000:
            # 近到連手續費都不夠，就抽掉
            fee = self.binance_maker_fee + self.bingx_taker_fee
            min_limit_cond = self.bingx_best_ask >= float(self.binance_now_order['price']) / (1+fee)
            # 遠到不太可能成交，就抽掉
            max_limit_cond = self.bingx_best_ask <= float(self.binance_now_order['price']) / (1+threshold*2)
            if min_limit_cond or max_limit_cond:
                self.logger.info(f"    ==> cancel order {self.binance_now_order['orderId']}")
                res = asyncio.run(self.binance_exchange.cancel_order(self.binance_symbol, self.binance_now_order['orderId']))
                self.binance_cancel_order_cooldown = int(time.time_ns() // 1000000) + self.binance_cancel_order_cooldown_ms
                if res.get('code', None) != None:
                    if res.get('code') == -2011: # failed: {"code": -2011, "msg": "Unknown order sent."}
                        self.binance_now_order = None
                    else:
                        self.logger.error(f"    => cancel order failed: {json.dumps(res)}")
        # 沒成交，有掛單，未超時 => 等待
        else:
            pass
        self.binance_mutax.release()

    def keyinterrupt(self):
        self.logger.info("Force stop, cancel all orders, all positions, and exit")
        # 清理未成交單
        orders = asyncio.run(self.binance_exchange.fetch_open_orders(self.binance_symbol))
        if isinstance(orders, list) and len(orders) > 0:
            for order in orders:
                self.logger.info(f"cancel order {order['orderId']}")
                asyncio.run(self.binance_exchange.cancel_order(self.binance_symbol, order['orderId']))

    def start(self):
        funcs = {
            "init": self.init,
            "wait_buy_in": self.wait_buy_in,
            "wait_close": self.wait_close,
            "keyinterrupt": self.keyinterrupt
        }
        while True:
            try:
                func = funcs[self.state] if funcs.get(self.state) else None
                if func:
                    func()
                else:
                    self.logger.error(f"Unknown state: {self.state}")
                    break
                if self.state == "keyinterrupt":
                    break
            except KeyboardInterrupt:
                self.state = "keyinterrupt"
            except Exception as e:
                self.logger.error(e)
                break

        self.logger.info("Bot stopped")
        self.logger.info("Closing sockets")
        self.binance_user_socket.ws.close()
        self.binance_orderbook_socket.ws.close()
        self.bingx_orderbook_socket.ws.close()
        self.logger.info("Exiting")
