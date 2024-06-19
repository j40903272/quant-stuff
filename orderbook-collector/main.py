import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
from datetime import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--symbol", type=str, default="BTC-USDT")
parser.add_argument("--platform", type=str, default="binance", choices=["binance", "bingx"])
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()

from pluto.utils.websocket import BinanceOrderbookSocket
from pluto.utils.websocket import BingxOrderbookWebsocket

def orderbook_to_txt_handler(datas):
    if not args.output:
        return

    dt = datetime.now().strftime('%Y%m%d%H')
    symbol = args.symbol
    platform = args.platform
    filename = f"{dt}-{symbol}-{platform}.txt"
    filepath = os.path.join(args.output, filename)
    with open(filepath, "a") as f:
        f.write(json.dumps(datas) + "\n")

if __name__ == '__main__':
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)

    symbol = args.symbol
    clazz = None
    if args.platform == "bingx":
        clazz = BingxOrderbookWebsocket
    elif args.platform == "binance":
        symbol = symbol.replace("-", "").lower()
        clazz = BinanceOrderbookSocket
    else:
        raise NotImplementedError

    websocket = clazz(symbol)
    websocket.handler = orderbook_to_txt_handler
    websocket.start()
