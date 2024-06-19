import argparse
import os
import json
import threading
import time
import datetime
import pandas as pd
import logging
import os
if not os.path.exists("logs"):
    os.makedirs("logs")
logger = logging.getLogger(__name__)

from Exchange.binance_exchange import BINANCE
from Exchange.bingx_exchange import BINGX

class MONITOR():

    def __init__(self, config):
        self.A_positions = 0
        self.B_positions = 0
        self.entry_long_prices = []
        self.entry_short_prices = []
        self.avg_long_price = 0
        self.avg_short_price = 0
        self.trade_records = []
        self.pnl = 0
        self.pnl_history = []
        self.binance_orderbook = {}
        self.bingx_orderbook = {}
 
        self.entry_threshold = 0.001
        self.binance_maker_cost = 0
        self.binance_taker_cost = 0.000153
        self.bingx_maker_cost = 0.0000045
        self.bingx_taker_cost = 0.000105
        self.want_to_earn = config["want_to_earn"]
        self.unit_size = 1000
        self.exit_threshold = self.entry_threshold - self.binance_maker_cost - self.bingx_taker_cost - self.want_to_earn # 出場時需要當下的持倉平均價差有無超過交易成本和我們想賺的錢
        self.unit = 1
        self.profit = 0
        
    def store_value(self, msg):
        if "exchange" in msg and msg["exchange"] == "binance": # binance
            if "e" in msg and msg["e"] == "bookTicker":
                self.binance_orderbook["ts"] = int(msg["T"])
                self.binance_orderbook["asks"] = float(msg["b"])
                self.binance_orderbook["bids"] = float(msg["a"])
        elif "exchange" in msg and msg["exchange"] == "bingx": # bingx
            if "depth" in msg["dataType"]:
                self.bingx_orderbook["ts"] = int(datetime.datetime.now().timestamp() * 1000)
                self.bingx_orderbook["asks"] = float(msg["data"]["asks"][0][0])
                self.bingx_orderbook["bids"] = float(msg["data"]["bids"][0][0])
        else:
            logger.critical("Uncatched msg:", msg)

    def spread_detector(self, binance_bid_1, binance_ask_1, bingx_bid_1, bingx_ask_1):
        
        now = datetime.datetime.now()

        binance_mid_price = (binance_bid_1 + binance_ask_1) / 2
        bingx_mid_price = (bingx_bid_1 + bingx_ask_1) / 2 

        # binance > bingx
        if binance_mid_price > bingx_mid_price: #long bingx short binance
            current_spread = (binance_mid_price - bingx_mid_price) / binance_mid_price
            if current_spread < self.exit_threshold and self.position == -1: # Close position
                logger.info("binance > bingx")
                logger.info(f"current binance price: {binance_mid_price}")
                logger.info(f"current bingx price: {bingx_mid_price}")
                logger.info(f"current_spread: {current_spread}")
                logger.info("spread < self.exit_threshold")
                logger.info("-------------------------------------")
                # Maker close Short: binance. Taker close Long: bingx.
                self.profit += (binance_bid_1 * self.binance_maker_cost + bingx_bid_1 * self.bingx_taker_cost) * -1 \
                            +  (self.open_position_price[0] - binance_bid_1) \
                            +  (self.open_position_price[1] - bingx_bid_1)
                self.position = 0
                # 'ts', 'binance_ask', 'binance_bid', 'bingx_ask', 'bingx_bid', 'spread', 'maker/maker', 'maker/taker', 'taker/taker', 'position', 'profit'
                self.df.loc[len(self.df)] = [now, binance_ask_1, binance_bid_1, bingx_ask_1, bingx_bid_1, current_spread, 0, 0, 0, self.position, self.profit]
                

            elif current_spread > self.entry_threshold and self.position != -1: # Open position
                maker_maker = round((binance_ask_1 - bingx_bid_1) / bingx_bid_1 * 100, 5)
                maker_taker = round((binance_bid_1 - bingx_bid_1) / bingx_bid_1 * 100, 5)
                taker_taker = round((binance_bid_1 - bingx_ask_1) / bingx_ask_1 * 100, 5)
                self.open_position_price = [binance_ask_1, bingx_ask_1] # Maker Short: binance. Taker Long: bingx.
                self.profit += (binance_ask_1 * self.binance_maker_cost + bingx_ask_1 * self.bingx_taker_cost) * -1
                self.position = -1
                logger.info("binance > bingx")
                logger.info(f"current binance price: {binance_mid_price}")
                logger.info(f"current bingx price: {bingx_mid_price}")
                logger.info(f"current_spread: {current_spread}")
                logger.info("spread > self.entry_threshold")
                logger.info(f"Long_Bingx_Short_Binance {current_spread} % |")
                logger.info(f"maker/maker {maker_maker} % |")
                logger.info(f"maker/taker {maker_taker} % |")
                logger.info(f"taker/taker {taker_taker} % |")
                logger.info("-------------------------")

                # 'ts', 'binance_ask', 'binance_bid', 'bingx_ask', 'bingx_bid', 'spread', 'maker/maker', 'maker/taker', 'taker/taker', 'position', 'profit'
                self.df.loc[len(self.df)] = [now, binance_ask_1, binance_bid_1, bingx_ask_1, bingx_bid_1, current_spread, maker_maker, maker_taker, taker_taker, self.position, self.profit]
                
        # bingx_bid_1 > binance_bid_1
        elif bingx_mid_price > binance_mid_price: #long binance short bingx
            current_spread = (bingx_mid_price - binance_mid_price) / bingx_mid_price
            if current_spread < self.exit_threshold and self.position == 1: # Close position
                logger.info("bingx > binance")
                logger.info(f"current binance price: {binance_mid_price}")
                logger.info(f"current bingx price: {bingx_mid_price}")
                logger.info(f"current_spread: {current_spread}")
                logger.info("spread < self.exit_threshold")
                logger.info("-------------------------------------")
                
                # Maker Close Long: binance. Taker Close Short: bingx.
                self.profit += (binance_ask_1 * self.binance_maker_cost + bingx_ask_1 * self.bingx_taker_cost) * -1 \
                            + ( self.open_position_price[0] - binance_ask_1) \
                            + ( self.open_position_price[1] - bingx_ask_1)
                self.position = 0
                self.df.loc[len(self.df)] = [now, binance_ask_1, binance_bid_1, bingx_ask_1, bingx_bid_1, current_spread, 0, 0, 0, self.position, self.profit]
                

            elif current_spread > self.entry_threshold and self.position != 1: # Open position
                maker_maker = round((bingx_ask_1 - binance_bid_1) / binance_bid_1 * 100, 5)
                maker_taker = round((bingx_ask_1 - binance_ask_1) / binance_ask_1 * 100, 5)
                taker_taker = round((bingx_bid_1 - binance_ask_1) / binance_ask_1 * 100, 5)
                self.open_position_price = [binance_bid_1, bingx_bid_1] # Maker Long: binance. Taker Short: bingx.
                self.profit += (binance_bid_1 * self.binance_maker_cost + bingx_bid_1 * self.bingx_taker_cost) * -1
                self.position = 1
                logger.info("bingx > binance")
                logger.info(f"current binance price: {binance_mid_price}")
                logger.info(f"current bingx price: {bingx_mid_price}")
                logger.info(f"current_spread: {current_spread}")
                logger.info("spread > self.entry_threshold")
                logger.info(f"Long_Binance_Short_Bingx {current_spread} % |")
                logger.info(f"maker/maker {maker_maker} % |")
                logger.info(f"taker/taker {taker_taker} % |")
                logger.info("--------------------------")
                
                self.df.loc[len(self.df)] = [now, binance_ask_1, binance_bid_1, bingx_ask_1, bingx_bid_1, current_spread, maker_maker, maker_taker, taker_taker, self.position, self.profit]
                    
    def run(self):
        self.df = pd.DataFrame(columns=['ts', 'binance_ask', 'binance_bid', 'bingx_ask', 'bingx_bid', 'spread', 'maker/maker', 'maker/taker', 'taker/taker', 'position', 'profit'])
        check = 0
        wait = 0
        self.position = 0
        # 0 = no position
        # 1 = long binance short bingx
        # -1 = long bingx short binance
        os.makedirs(f"./research/", exist_ok=True)
        while True:

            self.now_datetime = datetime.datetime.now()
            self.formatted_now = self.now_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            try:
                binance_ts = self.binance_orderbook["ts"]
                bingx_ts = self.bingx_orderbook["ts"]
                
            except Exception as e: # if orderbook is not updated yet
                logger.critical("Exception: " + str(e))
                time.sleep(1)
                wait += 1
                if wait > 10:
                    print("Lose connection")
                    os._exit(0)
                continue

            if abs(binance_ts - bingx_ts) <= 5000: # 5 secs
                binance_bid_1 = self.binance_orderbook["bids"]
                binance_ask_1 = self.binance_orderbook["asks"]
                bingx_bid_1 = self.bingx_orderbook["bids"]
                bingx_ask_1 = self.bingx_orderbook["asks"]

                self.spread_detector(binance_bid_1, binance_ask_1, bingx_bid_1, bingx_ask_1)

                if (self.now_datetime.minute == 0 and check == 0):
                    check = 1
                    self.df.to_csv(f'./research/{self.formatted_now}.csv', mode='a', index=False, header=False)
                    logger.info("df dumped!")
                    self.df = self.df.iloc[0:0] # clear dataframe

                if self.now_datetime.minute == 1:
                    check = 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="config path")
    parser.add_argument("-config", help="config", required=True)
    args = parser.parse_args()

    config_file_name = args.config
    with open(os.path.join("./", config_file_name)) as f:
        config = json.load(f)

    symbol = config["symbol"]
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s %(levelname)-8s %(funcName)s(): %(lineno)d %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f'logs/{symbol}.log',
                        filemode='w')
    binance = BINANCE(symbol)
    bingx = BINGX(symbol)

    monitor = MONITOR(config)
    threading.Thread(target=binance.start_ws, args=("bookTicker", monitor.store_value)).start()
    threading.Thread(target=bingx.start_ws, args=("depth5", monitor.store_value)).start()
    
    monitor.run()