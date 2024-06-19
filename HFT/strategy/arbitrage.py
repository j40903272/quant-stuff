import argparse
import configparser
import os
import json
import threading
import time
import datetime
import pandas as pd
import os

if not os.path.exists("logs"):
    os.makedirs("logs")

from const import Position_status
from Exchange.binance_exchange import BINANCE
from Exchange.bingx_exchange import BINGX
from utils import setup_logger
from binance.client import BinanceAPIException


class MONITOR:
    def __init__(self, config):
        api_key_ini = configparser.ConfigParser()
        api_key_ini.read(filenames=config["api_key_path"])

        self.symbol = config["symbol"]
        self.logger = setup_logger(
            self.symbol, is_verbose=True, is_file_handled=True, file_dir="logs"
        )
        self.binance = BINANCE(
            api_key_ini=api_key_ini["binance"],
            logger=self.logger,
            maker_cost=config["exchanges"]["binance"]["maker_cost"],
            taker_cost=config["exchanges"]["binance"]["taker_cost"],
            symbol=self.symbol,
        )
        self.bingx = BINGX(
            api_key_ini=api_key_ini["bingx"],
            logger=self.logger,
            maker_cost=config["exchanges"]["bingx"]["maker_cost"],
            taker_cost=config["exchanges"]["bingx"]["taker_cost"],
            symbol=self.symbol,
        )

        self.position_size_usdt = config["position_size_usdt"]
        self.want_to_earn = config["want_to_earn"]

        self.entry_threshold = (
            self.binance.maker_cost + self.bingx.taker_cost + self.want_to_earn
        )
        # 出場時需要當下的持倉平均價差有無超過交易成本和我們想賺的錢
        self.exit_threshold = (
            self.entry_threshold
            - self.binance.maker_cost
            - self.bingx.taker_cost
            - self.want_to_earn
        )

        self.profit = 0

        self.bingx_order_status = {}
        self.bingx_order_status["status"] = "NO_ORDER"

        self.position_status = Position_status.NO_POSITION

        self.logger.info(
            f"symbol: {self.symbol} \
                        position_size_usdt: {str(self.position_size_usdt)} \
                        binance.maker_cost: {str(self.binance.maker_cost)} \
                        binance.taker_cost: {str(self.binance.taker_cost)} \
                        bingx.maker_cost: {str(self.bingx.maker_cost)} \
                        bingx.taker_cost: {str(self.bingx.taker_cost)} \
                        want_to_earn: {str(self.want_to_earn)} \
                        entry_threshold: {str(self.entry_threshold)} \
                        exit_threshold: {str(self.exit_threshold)}"
        )

        for position in self.binance.get_all_position():
            if float(position["positionAmt"]) != 0.0 and position['symbol'] == self.symbol.upper() + "USDT":
                self.logger.info(f"Binance position: {str(position)}")
                raise ValueError("Binance position not 0")
        self.bingx.switch_leverage(leverage=1, side="LONG")
        self.bingx.switch_leverage(leverage=1, side="SHORT")

    def binance_user_ws_callback(self, msg):
        # binance ws {'e': 'ORDER_TRADE_UPDATE', 'T': 1695465193292, 'E': 1695465193295,
        # 'o': {'s': 'TRBUSDT', 'c': 'XjBMj8gJHozrf6qINCsscf', 'S': 'BUY', 'o': 'LIMIT',
        # 'f': 'GTC', 'q': '0.3', 'p': '34.500', 'ap': '34.5000', 'sp': '0', 'x': 'TRADE',
        # 'X': 'FILLED', 'i': 14565765586, 'l': '0.3', 'z': '0.3', 'L': '34.500', 'n': '0.00206999',
        # 'N': 'USDT', 'T': 1695465193292, 't': 252279749, 'b': '0', 'a': '10.3800', 'm': True,
        # 'R': False, 'wt': 'CONTRACT_PRICE', 'ot': 'LIMIT', 'ps': 'BOTH', 'cp': False, 'rp': '0',
        # 'pP': False, 'si': 0, 'ss': 0, 'V': 'NONE', 'pm': 'NONE', 'gtd': 0}}
        self.logger.info(f"binance ws {msg}")
        if msg["e"] == "ORDER_TRADE_UPDATE":
            if msg["o"]["S"] == "BUY":
                self.binance.long_order_status = msg
                self.binance.long_order_status["status"] = msg["o"]["X"]
            elif msg["o"]["S"] == "SELL":
                self.binance.short_order_status = msg
                self.binance.short_order_status["status"] = msg["o"]["X"]
                # self.logger.info(f"binance user ws: {str(msg)}")

    def bingx_user_ws_callback(self, msg):
        if msg["e"] == "ACCOUNT_CONFIG_UPDATE":
            return
        self.logger.info(f"bingx user ws: {str(msg)}")
        # if "orderId" in msg:
        #     self.bingx_order_status = msg
        #     self.logger.info(f"bingx user ws: {str(msg)}")

    def price_ws_callback(self, msg):
        # print(msg)
        if "exchange" in msg and msg["exchange"] == "binance":  # binance
            if "e" in msg and msg["e"] == "bookTicker":
                self.binance.orderbook["ts"] = int(msg["T"])
                self.binance.orderbook["asks"] = float(msg["b"])
                self.binance.orderbook["bids"] = float(msg["a"])

        elif "exchange" in msg and msg["exchange"] == "bingx":  # bingx
            if "depth" in msg["dataType"]:
                self.bingx.orderbook["ts"] = int(
                    datetime.datetime.now().timestamp() * 1000
                )
                self.bingx.orderbook["asks"] = float(msg["data"]["asks"][0][0])
                self.bingx.orderbook["bids"] = float(msg["data"]["bids"][0][0])
        else:
            self.logger.critical(f"Uncatched msg: {str(msg)}")

    def long_binance(self):
        if self.binance.long_order_status["status"] == "NO_ORDER":
            if self.position_status == Position_status.NO_POSITION:
                self.long_entry_price = self.bingx.ask_1 * (1 - self.entry_threshold)
            elif self.position_status == Position_status.LONG_BINGX_SHORT_BINANCE:
                self.long_entry_price = self.bingx.bid_1 * (1 + self.exit_threshold)
            else:
                self.logger.info("should not be here!")
                return

            if (
                self.long_entry_price >= self.binance.ask_1
                or self.long_entry_price == 0.0
                or self.binance.ask_1 == 0.0
            ):
                self.logger.info("binance long order not open!")
                return

            # self.logger.info(f"{self.position_status} long_entry_price: {self.long_entry_price}, binance.ask_1: {self.binance.ask_1}")
            try:
                self.long_quantity = round(
                    self.position_size_usdt / self.long_entry_price,
                    self.binance.quantityPrecision,
                )
                if self.long_quantity == 0:
                    raise Exception("long_quantity is 0.")
                response = self.binance.place_order(
                    quantity=self.long_quantity,
                    direction="LONG",
                    leverage=1,
                    price=self.long_entry_price,
                    stop_loss=None,
                    take_profit=None,
                    order_type="LIMIT",
                    time_in_force="GTX",
                )  # GTX = maker only

            except BinanceAPIException as e:
                print(str(e))
                if "APIError(code=-5022)" in str(e):  # Post only
                    print("binance long order post only!!!!")
                    return
            self.logger.info(
                f"{self.position_status} binance new long response: {str(response)}"
            )
            while (
                self.binance.long_order_status["status"] != "NEW"
            ):  # Due to latency of user websocket
                continue
            self.logger.info(
                f"{self.position_status} New Order binance.long_order_status['status']: {self.binance.long_order_status['status']}"
            )
            return

        elif self.binance.long_order_status["status"] == "NEW":
            if float(self.binance.long_order_status["o"]["p"]) >= self.bingx.ask_1 * (
                1 + self.entry_threshold * 2
            ):
                response = self.binance.cancel_order(
                    orderId=self.binance.long_order_status["o"]["i"]
                )
                self.logger.info(
                    f"{self.position_status} binance cancel long order response: {str(response)}"
                )
                while self.binance.long_order_status["status"] != "CANCELED":
                    self.logger.info(
                        f"{self.position_status} Wait for self.binance.long_order_status['status'] == 'CANCELED', now: {self.binance.long_order_status['status']}"
                    )
                    continue
                self.binance.long_order_status["status"] = "NO_ORDER"
            return

        elif (
            self.binance.long_order_status["status"] == "CANCELED"
        ):  # when the other side filled, and cancel order of this side.
            self.binance.long_order_status["status"] = "NO_ORDER"

        elif self.binance.long_order_status["status"] == "FILLED":
            self.binance.long_order_status["status"] = "NO_ORDER"
            self.logger.info(
                f"{self.position_status} binance.long_order_status['status']: {self.binance.long_order_status['status']}"
            )
            positionSide = (
                "SHORT"
                if self.position_status == Position_status.NO_POSITION
                else "LONG"
            )  # SELL SHORT = Open order, buy short = Close order

            self.logger.info(
                f"{self.position_status} bingx short market order positionSide: {positionSide}"
            )
            response = self.bingx.place_order(
                quantity=self.long_quantity,
                direction="SELL",
                positionSide=positionSide,
                leverage=1,
                order_type="MARKET",
            )
            if self.position_status == Position_status.NO_POSITION:
                self.position_status = Position_status.LONG_BINANCE_SHORT_BINGX
            else:
                self.position_status = Position_status.NO_POSITION
            self.logger.info(
                f"{self.position_status} bingx short market order response: {str(response)}"
            )
            self.logger.info(
                f"{self.position_status} binance.short_order_status['status']: {self.binance.short_order_status['status']}"
            )
            if self.binance.short_order_status["status"] == "NEW":
                response = self.binance.cancel_order(
                    orderId=self.binance.short_order_status["o"]["i"]
                )
                self.binance.short_order_status["status"] = "NO_ORDER"
                self.logger.info(
                    f"{self.position_status} binance cancel short order by long function, response: {str(response)}"
                )

            return

        elif self.binance.long_order_status["status"] == "PARTIALLY_FILLED":
            # self.bingx.place_order(position_size_usdt=self.position_size_usdt,
            #                         direction="SHORT",
            #                         leverage=1,
            #                         stop_loss=None,
            #                         take_profit=None,
            #                         order_type="MARKET")
            # self.position_status = Position_status.LONG_BINANCE_SHORT_BINGX
            self.logger.info(
                f"{self.position_status} bingx short PARTIALLY market order response."
            )
            return

        self.logger.info(
            f"{self.position_status} self.binance.long_order_status['status'] = {self.binance.long_order_status['status']}"
        )

    def short_binance(self):
        if self.binance.short_order_status["status"] == "NO_ORDER":
            if self.position_status == Position_status.NO_POSITION:
                self.short_entry_price = self.bingx.bid_1 * (1 + self.entry_threshold)
            elif self.position_status == Position_status.LONG_BINANCE_SHORT_BINGX:
                self.short_entry_price = self.bingx.ask_1 * (1 - self.exit_threshold)
            else:
                self.logger.info("should not be here!")
                return

            if (
                self.short_entry_price <= self.binance.bid_1
                or self.short_entry_price == 0.0
                or self.binance.bid_1 == 0.0
            ):
                self.logger.info("binance short order not open!")
                return

            self.logger.info(
                f"{self.position_status} short_entry_price: {self.short_entry_price}, binance.bid_1: {self.binance.bid_1}"
            )
            try:
                self.short_quantity = round(
                    self.position_size_usdt / self.short_entry_price,
                    self.binance.quantityPrecision,
                )
                if self.short_quantity == 0:
                    raise Exception("short_quantity is 0.")
                response = self.binance.place_order(
                    quantity=self.short_quantity,
                    direction="SHORT",
                    leverage=1,
                    price=self.short_entry_price,
                    stop_loss=None,
                    take_profit=None,
                    order_type="LIMIT",
                    time_in_force="GTX",
                )  # GTX = maker only
            except BinanceAPIException as e:
                print(str(e))
                if "APIError(code=-5022)" in str(e):  # Post only
                    print("binance short order post only!!!!")
                    return
            self.logger.info(
                f"{self.position_status} binance new short response: {str(response)}"
            )
            while (
                self.binance.short_order_status["status"] != "NEW"
            ):  # Due to latency of user websocket
                continue
            self.logger.info(
                f"{self.position_status} NEW ORDER binance.short_order_status['status']: {self.binance.short_order_status['status']}"
            )
            return

        elif self.binance.short_order_status["status"] == "NEW":
            if float(self.binance.short_order_status["o"]["p"]) >= self.bingx.bid_1 * (
                1 + self.entry_threshold * 2
            ):
                response = self.binance.cancel_order(
                    orderId=self.binance.short_order_status["o"]["i"]
                )
                self.logger.info(
                    f"{self.position_status} binance cancel short order response: {str(response)}"
                )
                while self.binance.short_order_status["status"] != "CANCELED":
                    self.logger.info(
                        f"{self.position_status} Wait for self.binance.short_order_status['status'] == 'CANCELED', now: {self.binance.short_order_status['status']}"
                    )
                    continue
                self.binance.short_order_status["status"] = "NO_ORDER"
                self.logger.info(
                    f"{self.position_status} CANCELED ORDER binance.short_order_status['status']: {self.binance.short_order_status['status']}"
                )
            return

        elif (
            self.binance.short_order_status["status"] == "CANCELED"
        ):  # when the other side filled, and cancel order of this side.
            self.binance.short_order_status["status"] = "NO_ORDER"

        elif self.binance.short_order_status["status"] == "FILLED":
            self.binance.short_order_status["status"] = "NO_ORDER"
            self.logger.info(
                f"{self.position_status} binance.short_order_status['status']: {self.binance.short_order_status['status']}"
            )
            positionSide = (
                "LONG"
                if self.position_status == Position_status.NO_POSITION
                else "SHORT"
            )
            self.logger.info(
                f"{self.position_status} bingx long market order positionSide: {positionSide}"
            )
            response = self.bingx.place_order(
                quantity=self.short_quantity,
                direction="BUY",
                positionSide=positionSide,
                leverage=1,
                order_type="MARKET",
            )
            if self.position_status == Position_status.NO_POSITION:
                self.position_status = Position_status.LONG_BINGX_SHORT_BINANCE
            else:
                self.position_status = Position_status.NO_POSITION
            self.logger.info(
                f"{self.position_status} bingx long market order response: {str(response)}"
            )
            self.logger.info(
                f"{self.position_status} binance.long_order_status['status']: {self.binance.long_order_status['status']}"
            )
            if self.binance.long_order_status["status"] == "NEW":
                response = self.binance.cancel_order(
                    orderId=self.binance.long_order_status["o"]["i"]
                )
                self.binance.long_order_status["status"] = "NO_ORDER"
                self.logger.info(
                    f"{self.position_status} binance cancel long order by short function, response: {str(response)}"
                )
            return

        elif self.binance.short_order_status["status"] == "PARTIALLY_FILLED":
            # self.bingx.place_order(position_size_usdt=self.position_size_usdt,
            #                         direction="SHORT",
            #                         leverage=1,
            #                         take_profit=None,
            #                         order_type="MARKET")
            # self.position_status = Position_status.LONG_BINANCE_SHORT_BINGX
            self.logger.info(
                f"{self.position_status} bingx long PARTIALLY market order response."
            )
            return
        self.logger.info(
            f"{self.position_status} self.binance.short_order_status['status'] = {self.binance.short_order_status['status']}"
        )

    def run_long_binance_short_bingx(self):
        while True:
            if self.position_status == Position_status.NO_POSITION:
                self.long_binance()
            elif self.position_status == Position_status.LONG_BINANCE_SHORT_BINGX:
                self.short_binance()

    def run_long_bingx_short_binance(self):
        while True:
            if self.position_status == Position_status.NO_POSITION:
                self.short_binance()
            elif self.position_status == Position_status.LONG_BINGX_SHORT_BINANCE:
                self.long_binance()

    def get_price(self):
        # self.df = pd.DataFrame(columns=['ts', 'binance_ask', 'binance_bid', 'bingx_ask', 'bingx_bid', 'spread', 'position', 'profit'])
        # check = 0
        wait = 0
        os.makedirs(f"./research/", exist_ok=True)
        while True:
            try:
                binance_ts = self.binance.orderbook["ts"]
                bingx_ts = self.bingx.orderbook["ts"]

            except Exception as e:  # if orderbook is not updated yet
                self.logger.critical(f"Exception: {str(e)}")
                time.sleep(1)
                wait += 1
                if wait > 10:
                    self.logger.info("Lose connection")
                    os._exit(0)
                continue

            if abs(binance_ts - bingx_ts) <= 5000:  # 5 secs
                self.binance.bid_1 = self.binance.orderbook["bids"]
                self.binance.ask_1 = self.binance.orderbook["asks"]
                self.bingx.bid_1 = self.bingx.orderbook["bids"]
                self.bingx.ask_1 = self.bingx.orderbook["asks"]

                # self.now_datetime = datetime.datetime.now()
                # self.formatted_now = self.now_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                # if (self.now_datetime.minute == 0 and check == 0):
                #     check = 1
                # self.df.to_csv(f'./research/{self.formatted_now}.csv', mode='a', index=False, header=False)
                # self.logger.info("df dumped!")
                # self.df = self.df.iloc[0:0] # clear dataframe

                # if self.now_datetime.minute == 1:
                #     check = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config path")
    parser.add_argument("-config", help="config", required=True)
    args = parser.parse_args()

    config_file_name = args.config
    with open(os.path.join("./", config_file_name)) as f:
        config = json.load(f)

    monitor = MONITOR(config)

    threading.Thread(
        target=monitor.binance.start_ws,
        args=(
            "bookTicker",
            monitor.price_ws_callback,
        ),
    ).start()
    threading.Thread(
        target=monitor.bingx.start_ws,
        args=(
            "depth5",
            monitor.price_ws_callback,
        ),
    ).start()

    threading.Thread(
        target=monitor.binance.start_user_ws, args=(monitor.binance_user_ws_callback,)
    ).start()
    threading.Thread(
        target=monitor.bingx.start_ws,
        args=(
            "user_ws",
            monitor.bingx_user_ws_callback,
        ),
    ).start()

    threading.Thread(target=monitor.run_long_binance_short_bingx).start()
    threading.Thread(target=monitor.run_long_bingx_short_binance).start()

    monitor.get_price()
