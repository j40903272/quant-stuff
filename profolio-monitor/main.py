from dotenv import load_dotenv
load_dotenv()
import os
import sys
sys.path.append(os.getcwd())

import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from pluto.exchange import BinanceExchange, BingxExchange

import asyncio
import datetime
import pandas as pd

token = os.environ.get("INFLUXDB_TOKEN")
bucket = "test"
org = "alexandex"
url = "http://18.183.164.243:8086"
uuid = "59bfef17-11f0-4e20-a19e-b7d5692907f6"

client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

binance_exchange = BinanceExchange(os.getenv('BINANCE_APIKEY'), os.getenv('BINANCE_SECRET'))
bingx_exchange = BingxExchange(os.getenv('BINGX_APIKEY'), os.getenv('BINGX_SECRET'))

async def push_balance():
    binance_balance = await binance_exchange.get_balance()
    bingx_balance = await bingx_exchange.get_balance()
    print("Binance balance: ", binance_balance)
    print("Bingx balance: ", bingx_balance)

    point = Point("account").tag("uuid", uuid).tag("exchange", "binance").field("balance", binance_balance)
    write_api.write(bucket, org, point)
    point = Point("account").tag("uuid", uuid).tag("exchange", "bingx").field("balance", bingx_balance)
    write_api.write(bucket, org, point)

async def push_margin():
    binance_margin = await binance_exchange.get_available_balance()
    bingx_margin = await bingx_exchange.get_available_balance()
    print("Binance margin: ", binance_margin)
    print("Bingx margin: ", bingx_margin)

    point = Point("account").tag("uuid", uuid).tag("exchange", "binance").field("margin", binance_margin)
    write_api.write(bucket, org, point)
    point = Point("account").tag("uuid", uuid).tag("exchange", "bingx").field("margin", bingx_margin)
    write_api.write(bucket, org, point)

async def get_profolio_df(symbol):
    df = pd.DataFrame(columns=['time', 'orderId', 'symbol', 'exchange', 'side', 'price', 'qty', 'profit', 'commission'])
    binance_trades = await binance_exchange.fetch_trade_list(symbol.replace("-", ""))
    if len(binance_trades) > 0:
        binance_df = pd.DataFrame(binance_trades)
        binance_df['exchange'] = ['binance' for _ in range(len(binance_df))]
        binance_df['symbol'] = [symbol for _ in range(len(binance_df))]
        binance_df['profit'] = binance_df['realizedPnl']
        binance_df['commission'] = binance_df['commission'].astype(float)
        binance_df['commission'] = binance_df['commission'] * -1.0
        df = df._append(binance_df[['time', 'orderId', 'symbol', 'exchange', 'side', 'price', 'qty', 'profit', 'commission']])

    bingx_trades = await bingx_exchange.history_orders(symbol, (time.time() - 4 * 60 * 60) * 1000, 1000)
    if len(bingx_trades) > 0:
        bingx_df = pd.DataFrame(bingx_trades)
        bingx_df['exchange'] = ['bingx' for _ in range(len(bingx_df))]
        bingx_df['qty'] = bingx_df['origQty'].astype(float)
        bingx_df['symbol'] = [symbol for _ in range(len(bingx_df))]
        bingx_df['price'] = bingx_df['avgPrice'] # Bingx Market Order
        df = df._append(bingx_df[['time', 'orderId', 'symbol', 'exchange', 'side', 'price', 'qty', 'profit', 'commission']])


    df['time'] = pd.to_datetime(df['time'], unit='ms')
    one_hour_ago = pd.Timestamp.now() - pd.Timedelta(hours=24)
    df = df[df["time"] >= one_hour_ago]
    df = df.sort_values(by='time')
    df['profit'] = df['profit'].astype(float)
    df['commission'] = df['commission'].astype(float)
    df['qty'] = df['qty'].astype(float)
    df['price'] = df['price'].astype(float)
    df['real_profit'] = df['profit'] + df['commission']
    df['trade_count'] = 1
    df['trade_volume'] = df['price'] * df['qty']
    # group by YYYY-MM-DD-HH and sum real_profit
    df['time'] = df['time'].dt.strftime('%Y-%m-%d-%H-%M-%S')
    results = df[['time', 'symbol', 'exchange', 'profit', 'commission', 'real_profit', 'trade_count', 'trade_volume']].groupby(['time', 'symbol', 'exchange']).sum()
    results = results.reset_index()
    results.sort_values(by='time')

    return results
    
jobs = ["dual-spread", "dual-spread", "dual-spread", "dual-spread", "dual-spread", "dual-spread", "dual-spread", "latancy-taker", "latancy-taker", "orderbook-ml-taker", "CTA", "Spot-perp", "Grid", "Grid"]
symbols = ["WIF-USDT", "DOGE-USDT", "POLYX-USDT", "ETHFI-USDT", "AVAX-USDT", "ENA-USDT", "SOL-USDT", "WLD-USDT", "SUI-USDT", "BTC-USDT", "BNB-USDT", "STX-USDT", "ETH-USDT", "IOTA-USDT"]
symbol_index = 0
async def push_profolio():
    global symbols, symbol_index, jobs
    job = jobs[symbol_index]
    symbol = symbols[symbol_index]
    symbol_index = (symbol_index + 1) % (len(symbols))
    df = await get_profolio_df(symbol)
    df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d-%H-%M-%S")
    print("now symbol", symbol)
    points = []
    for index, row in df.iterrows():
        point = (Point("pnl")
                    .tag("uuid", uuid)
                    .tag("exchange", row["exchange"])
                    .tag("symbol", symbol)
                    .tag("job", job)
                    .time(row["time"])
                    .field("profit", row["profit"]))
        points.append(point)
        point = (Point("pnl")
                    .tag("uuid", uuid)
                    .tag("exchange", row["exchange"])
                    .tag("symbol", symbol)
                    .tag("job", job)
                    .time(row["time"])
                    .field("commission", row["commission"]))
        points.append(point)
        point = (Point("pnl")
                    .tag("uuid", uuid)
                    .tag("exchange", row["exchange"])
                    .tag("symbol", symbol)
                    .tag("job", job)
                    .time(row["time"])
                    .field("real_profit", row["real_profit"]))
        points.append(point)
        point = (Point("pnl")
                    .tag("uuid", uuid)
                    .tag("exchange", row["exchange"])
                    .tag("symbol", symbol)
                    .tag("job", job)
                    .time(row["time"])
                    .field("trade_count", row["trade_count"]))
        points.append(point)
        point = (Point("pnl")
                    .tag("uuid", uuid)
                    .tag("exchange", row["exchange"])
                    .tag("symbol", symbol)
                    .tag("job", job)
                    .time(row["time"])
                    .field("trade_volume", row["trade_volume"]))
        points.append(point)
    print("add", len(points), "data points")
    write_api.write(bucket, org, points)

# 餘額
async def main():
    try:
        await push_balance()
    except:
        time.sleep(55)
        pass
    time.sleep(5)
    
    try:
        await push_margin()
    except:
        time.sleep(55)
        pass
    time.sleep(5)

    try:
        await push_profolio()
    except:
        time.sleep(55)
        pass
    time.sleep(5)

if __name__ == "__main__":
    while True:
        asyncio.run(main())
