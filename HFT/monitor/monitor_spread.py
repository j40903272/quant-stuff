symbols = ['KSMUSDT', 'TLMUSDT', 'LTCUSDT', 'AGIXUSDT', 'NEARUSDT', 'ALGOUSDT', 'CHRUSDT', 'UNIUSDT', 'ASTRUSDT', 'GMXUSDT', 'COTIUSDT', 'MANAUSDT', 'TWTUSDT', 'DUSKUSDT', 'LDOUSDT', 'GALAUSDT', 'SOLUSDT', 'UNFIUSDT', 'CELRUSDT', 'ATOMUSDT', 'ZRXUSDT', 'KNCUSDT', 'SEIUSDT', 'SUSHIUSDT', 'MINAUSDT', 'INJUSDT', 'KEYUSDT', 'CRVUSDT', 'ZENUSDT', 'ARBUSDT', 'WLDUSDT', 'PENDLEUSDT', 'OCEANUSDT', 'GMTUSDT', 'NEOUSDT', 'GALUSDT', 'LQTYUSDT', 'CHZUSDT', 'NMRUSDT', 'MKRUSDT', 'RNDRUSDT', 'LPTUSDT', 'RLCUSDT', 'COMPUSDT', 'PERPUSDT', 'TRXUSDT', 'DOTUSDT', 'PEOPLEUSDT', 'ACHUSDT', 'CFXUSDT', 'API3USDT', '1INCHUSDT', 'RUNEUSDT', 'QNTUSDT', 'WOOUSDT', 'DOGEUSDT', 'ATAUSDT', 'YFIUSDT', 'BNTUSDT', 'LITUSDT', 'IDUSDT', 'JASMYUSDT', 'OPUSDT', 'AXSUSDT', 'MATICUSDT', 'KAVAUSDT', 'PEPEUSDT', 'OXTUSDT', 'XEMUSDT', 'IOSTUSDT', 'FILUSDT', 'FTMUSDT', 'IDEXUSDT', 'DYDXUSDT', 'SHIBUSDT', 'JOEUSDT', 'AUDIOUSDT', 'ETCUSDT', 'TOMOUSDT', 'TRBUSDT', 'ARPAUSDT', 'RADUSDT', 'YGGUSDT', 'ZILUSDT', 'UMAUSDT', 'BCHUSDT', 'KLAYUSDT', 'AGLDUSDT', 'THETAUSDT', 'ONEUSDT', 'CTKUSDT', 'HOOKUSDT', 'SFPUSDT', 'XMRUSDT', 'LRCUSDT', 'XLMUSDT', 'STMXUSDT', 'ENJUSDT', 'LINKUSDT', 'LUNCUSDT', 'OMGUSDT', 'BNXUSDT', 'ADAUSDT', 'BNBUSDT', 'BATUSDT', 'FLOKIUSDT', 'BLZUSDT', 'WAVESUSDT', 'USTCUSDT', 'ARKMUSDT', 'AAVEUSDT', 'EDUUSDT', 'CELOUSDT', 'MAGICUSDT', 'PHBUSDT', 'MAVUSDT', 'MASKUSDT', 'SUIUSDT', 'STGUSDT', 'XRPUSDT', 'STXUSDT', 'VETUSDT', 'APEUSDT', 'HBARUSDT', 'SANDUSDT', 'GTCUSDT', 'TRUUSDT', 'IMXUSDT', 'USDCUSDT', 'DODOUSDT', 'ICXUSDT', 'RVNUSDT', 'ENSUSDT', 'FXSUSDT', 'MDTUSDT', 'LUNAUSDT', 'SNXUSDT', 'FLOWUSDT', 'LINAUSDT', 'COMBOUSDT', 'DARUSDT', 'BTCUSDT', 'HIGHUSDT', 'RENUSDT', 'OGNUSDT', 'ONTUSDT', 'SLPUSDT', 'REEFUSDT', 'AVAXUSDT', 'ICPUSDT', 'FETUSDT', 'STORJUSDT', 'ANKRUSDT', 'ETHUSDT', 'HOTUSDT', 'EOSUSDT', 'IOTAUSDT', 'GRTUSDT', 'SSVUSDT', 'CYBERUSDT', 'FLMUSDT', 'EGLDUSDT', 'APTUSDT', 'ROSEUSDT', 'SXPUSDT']
discord_hook_url = "xxx"
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import asyncio
import logging
import threading
from datetime import datetime

import requests
import numpy as np
import pandas as pd

from src.websocket.BingXSocket import BingXSocket
from src.websocket.BinanceFutureSocket import BinanceFutureSocket

import redis

logger = logging.getLogger(__name__)

dn = {}
std = {}
mean = {}

for s in symbols:
    dn[s] = 0
    std[s] = 0
    mean[s] = 0

binance_t = {}
binance_bid = {}
binance_ask = {}
bingx_t = {}
bingx_bid = {}
bingx_ask = {}

async def monitor():
    global dn, std, mean, records, symbols, binance_t, binance_bid, binance_ask, bingx_t, bingx_bid, bingx_ask
    nowdatetime = datetime.now().strftime('%Y%m%d%H')
    r = redis.Redis(host='localhost', port=6379, db=0)

    while True:
        await asyncio.sleep(.1)
        stime = time.time_ns() // 10**6
        for s in symbols:
            if binance_t.get(s) is None or bingx_t.get(s) is None:
                continue
            current_price = sum([bingx_bid.get(s), bingx_ask.get(s), binance_bid.get(s), binance_ask.get(s)]) / 4
            spread = max(abs(bingx_bid.get(s) - binance_bid.get(s)), abs(bingx_ask.get(s) - binance_ask.get(s))) / current_price
            dn[s] += 1
            std[s] = np.sqrt(((dn[s] - 1) * std[s]**2 + (spread - mean[s])**2) / dn[s])
            mean[s] = ((dn[s] - 1) * mean[s] + spread) / dn[s]

        connection_finish_time = time.time_ns() // 10**6

        # if change to next hour, reset records & save to file
        if datetime.now().strftime('%S') == '00':
            stime = time.time_ns() // 10**6
            spread_msg = ""
            for i in range(1, 11):
                r.publish(
                    f'spread{i}',
                    spread_msg,
                )
            df = pd.DataFrame()
            df.insert(0, 'symbol', symbols)
            df.insert(1, 'spread_std', [std[s] for s in symbols])
            df.insert(2, 'spread_mean', [mean[s] for s in symbols])
            df = df.sort_values(by=['spread_std'], ascending=False)
            print(df.head(10))
            print("Cost time:", time.time_ns() // 10**6 - stime, "ms")
            time.sleep(1)

def bingx_handler(message):
    global bingx_t, bingx_bid, bingx_ask
    j = json.loads(message)
    if j and not j.get('dataType', '').endswith('@depth5'):
        return
    s = j['dataType'].split('@')[0].replace("-", "").upper()
    bingx_t[s] = time.time_ns() // 10**6
    bingx_bid[s] = float(j['data']['bids'][0][0])
    bingx_ask[s] = float(j['data']['asks'][-1][0])

def binance_handler(message):
    global binance_t, binance_bid, binance_ask
    j = json.loads(message)
    if j and j.get('e') != 'bookTicker':
        return
    s = j['s'].replace("-", "").upper()
    binance_t[s] = time.time_ns() // 10**6
    binance_bid[s] = float(j['b'])
    binance_ask[s] = float(j['a'])

if __name__ == "__main__":
    binance_channel = list(map(lambda x: x.lower() + "@bookTicker", symbols))
    bfw = BinanceFutureSocket(channels=binance_channel)
    bfw.handler = binance_handler
    threading.Thread(target=bfw.start).start()

    bingx_channel = list(map(lambda x: x.replace("USDT", "-USDT") + "@depth5", symbols))
    bingx = BingXSocket(channels=bingx_channel)
    bingx.handler = bingx_handler
    threading.Thread(target=bingx.start).start()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(monitor())
    loop.close()