from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.append(os.getcwd())
import time
import asyncio
import datetime
import pandas as pd
# set dataframe width
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

from pluto.exchange import BinanceExchange, BingxExchange

async def main():
    # This will cause error, comment it out
    # symbols = ['REN-USDT', 'LTC-USDT', 'ARB-USDT', 'LDO-USDT', 'SUI-USDT', 'XRP-USDT', 'NEO-USDT']
    symbols = ['NEO-USDT']

    binance_exchange = BinanceExchange(os.getenv('BINANCE_APIKEY'), os.getenv('BINANCE_SECRET'))
    bingx_exchange = BingxExchange(os.getenv('BINGX_APIKEY'), os.getenv('BINGX_SECRET'))

    df = pd.DataFrame(columns=['time', 'orderId', 'symbol', 'exchange', 'side', 'price', 'qty', 'profit', 'commission'])
    for symbol in symbols:
        binance_trades = await binance_exchange.fetch_trade_list(symbol.replace("-", ""))
        binance_df = pd.DataFrame(binance_trades)
        binance_df['exchange'] = ['binance' for _ in range(len(binance_df))]
        binance_df['symbol'] = [symbol for _ in range(len(binance_df))]
        binance_df['profit'] = binance_df['realizedPnl']
        bingx_trades = await bingx_exchange.fetch_trade_list(symbol)
        bingx_df = pd.DataFrame(bingx_trades)
        bingx_df['exchange'] = ['bingx' for _ in range(len(bingx_df))]
        bingx_df['qty'] = bingx_df['origQty'].astype(float)
        bingx_df['symbol'] = [symbol for _ in range(len(bingx_df))]

        df = df._append(binance_df[['time', 'orderId', 'symbol', 'exchange', 'side', 'price', 'qty', 'profit', 'commission']])
        df = df._append(bingx_df[['time', 'orderId', 'symbol', 'exchange', 'side', 'price', 'qty', 'profit', 'commission']])
        time.sleep(1)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.sort_values(by='time')
    print(df)
    df['profit'] = df['profit'].astype(float)
    df['commission'] = df['commission'].astype(float)
    df['qty'] = df['qty'].astype(float)
    df['price'] = df['price'].astype(float)
    df['real_profit'] = df['profit'] + df['commission']
    df['trade_count'] = 1
    df['trade_volume'] = df['price'] * df['qty']
    # group by YYYY-MM-DD-HH and sum real_profit
    df['time'] = df['time'].dt.strftime('%Y-%m-%d-%H')
    results = df[['time', 'symbol', 'real_profit', 'trade_count', 'trade_volume']].groupby(['time', 'symbol']).sum()
    results = results.reset_index()
    results.sort_values(by='time')
    print(results.tail(12))
    df.to_csv('data.csv')

asyncio.run(main())