from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.append(os.getcwd())
import asyncio
from pluto.exchange import BinanceExchange, BingxExchange

import pandas as pd
from prometheus_client import Counter, CollectorRegistry, Gauge, start_http_server

from ratelimit import limits, sleep_and_retry

registry = CollectorRegistry()
balance_gauge = Gauge('balance', 'Balance', ['exchange', 'currency'], registry=registry)
position_gauge = Gauge('position', 'Position', ['exchange', 'symbol', 'positionSide', 'leverage'], registry=registry)

@sleep_and_retry
@limits(calls=1, period=10)
async def main():
    binance_exchange = BinanceExchange(os.getenv('BINANCE_APIKEY'), os.getenv('BINANCE_SECRET'))
    bingx_exchange = BingxExchange(os.getenv('BINGX_APIKEY'), os.getenv('BINGX_SECRET'))

    # binance_balance = await binance_exchange.get_balance()


    columns = ['exchange', 'symbol', 'positionAmt', 'unRealizedProfit', 'positionSide', 'leverage']
    positions_df = pd.DataFrame(columns=columns)
    binance_positions = await binance_exchange.fetch_position("")
    binance_positions = pd.DataFrame(binance_positions)
    binance_positions['exchange'] = 'binance'
    binance_positions['positionAmt'] = binance_positions['positionAmt'].astype(float)
    binance_positions['unRealizedProfit'] = binance_positions['unRealizedProfit'].astype(float)
    binance_positions['leverage'] = binance_positions['leverage'].astype(float)
    binance_positions = binance_positions[columns]
    # filter out empty positions
    binance_positions = binance_positions[binance_positions['positionAmt'] != 0]
    print(binance_positions)
    positions_df = positions_df.append(binance_positions, ignore_index=True)

    bingx_positions = await bingx_exchange.fetch_position("")
    if bingx_positions:
        bingx_positions = pd.DataFrame(bingx_positions)
        bingx_positions['exchange'] = 'bingx'
        bingx_positions['positionAmt'] = bingx_positions['positionAmt'].astype(float)
        bingx_positions['unRealizedProfit'] = bingx_positions['unRealizedProfit'].astype(float)
        bingx_positions['leverage'] = bingx_positions['leverage'].astype(float)
        bingx_positions = bingx_positions[columns]
        bingx_positions = bingx_positions[bingx_positions['positionAmt'] != 0]
        print(bingx_positions)
        positions_df = positions_df.append(bingx_positions, ignore_index=True)

    for index, row in positions_df.iterrows():
        position_gauge.labels(row['exchange'], row['symbol'], row['positionSide'], row['leverage']).set(row['positionAmt'])

    binance_balance = await binance_exchange.get_balance()
    bingx_balance = await bingx_exchange.get_balance()
    print("Binance balance: ", binance_balance)
    print("Bingx balance: ", bingx_balance)
    balance_gauge.labels('binance', 'USDT').set(binance_balance)
    balance_gauge.labels('bingx', 'USDT').set(bingx_balance)

if __name__ == "__main__":
    start_http_server(8000, registry=registry)
    while True:
        asyncio.run(main())