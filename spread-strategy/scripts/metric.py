from dotenv import load_dotenv
load_dotenv()
import os
import sys
sys.path.append(os.getcwd())
import asyncio

import glob
import os
import time
import re

from ratelimit import limits, sleep_and_retry

import pandas as pd

from pluto.exchange import BinanceExchange, BingxExchange
binance_exchange = BinanceExchange(os.getenv('BINANCE_APIKEY'), os.getenv('BINANCE_SECRET'))
bingx_exchange = BingxExchange(os.getenv('BINGX_APIKEY'), os.getenv('BINGX_SECRET'))

@sleep_and_retry
@limits(calls=3, period=10)
async def get_binance_profolio_df(symbol):
    binance_trades = await binance_exchange.fetch_trade_list(symbol.replace("-", ""))
    binance_df = pd.DataFrame(binance_trades)
    binance_df['exchange'] = ['binance' for _ in range(len(binance_df))]
    binance_df['symbol'] = [symbol for _ in range(len(binance_df))]
    binance_df['profit'] = binance_df['realizedPnl']
    binance_df = binance_df[['time', 'orderId', 'symbol', 'exchange', 'side', 'price', 'qty', 'profit', 'commission']]
    binance_df['profit'] = binance_df['profit'].astype(float)
    binance_df['commission'] = binance_df['commission'].astype(float)
    binance_df['qty'] = binance_df['qty'].astype(float)
    binance_df['price'] = binance_df['price'].astype(float)
    return binance_df

@sleep_and_retry
@limits(calls=3, period=10)
async def get_bingx_profolio_df(symbol):
    bingx_trades = await bingx_exchange.fetch_trade_list(symbol)
    bingx_df = pd.DataFrame(bingx_trades)
    bingx_df['exchange'] = ['bingx' for _ in range(len(bingx_df))]
    bingx_df['qty'] = bingx_df['origQty'].astype(float)
    bingx_df['symbol'] = [symbol for _ in range(len(bingx_df))]
    bingx_df = bingx_df[['time', 'orderId', 'symbol', 'exchange', 'side', 'price', 'qty', 'profit', 'commission']]
    bingx_df['profit'] = bingx_df['profit'].astype(float)
    bingx_df['commission'] = bingx_df['commission'].astype(float)
    bingx_df['qty'] = bingx_df['qty'].astype(float)
    bingx_df['price'] = bingx_df['price'].astype(float)
    return bingx_df

from prometheus_client import Counter, CollectorRegistry, Gauge, generate_latest, start_http_server
order_counter = Counter('log_action_counter', 'Counter for log actions', ['action', 'symbol'])
profit_gauge = Gauge('profit_gauge', 'Gauge for profit', ['symbol', 'type', 'exchange'])
heartbeat_gauge = Gauge('heartbeat_gauge', 'Gauge for heartbeat')
balance_gauge = Gauge('balance_gauge', 'Gauge for balance', ['symbol', 'exchange'])
registry = CollectorRegistry()
registry.register(order_counter)
registry.register(profit_gauge)
registry.register(heartbeat_gauge)
registry.register(balance_gauge)

def push_to_gateway(gateway, job, registry):
    import requests
    data = generate_latest(registry)

    url = f"{gateway}/metrics/job/{job}"
    res = requests.post(url, data=data)
    if res.status_code != 200:
        raise Exception("推送失敗。")

def handler(parsed_log):    
    symbol = parsed_log['logger_name'].split('.')[0]
    binance_filled_pattern = re.compile(r'\(ws\): Order (\d+) filled')
    binance_fiiled_match = binance_filled_pattern.match(parsed_log['message'])
    bingx_filled_pattern = re.compile(r'BingX take market order done, order id: (\d+)')
    bingx_fiiled_match = bingx_filled_pattern.match(parsed_log['message'])
    if 'place limit order' in parsed_log['message']:
        order_counter.labels(action='place_limit_order', symbol=symbol).inc()
        print(symbol, 'place limit order')
    elif 'cancel order' in parsed_log['message']:
        order_counter.labels(action='cancel_order', symbol=symbol).inc()
        print(symbol, 'cancel order')
    elif binance_fiiled_match:
        order_id = binance_fiiled_match.group(1)
        df = asyncio.run(get_binance_profolio_df(symbol))
        order = df[df['orderId'] == int(order_id)]
        profit = order['profit'].sum()
        commission = order['commission'].sum()
        last_profit = profit_gauge.labels(symbol=symbol, type='profit', exchange='binance')._value.get()
        last_commission = profit_gauge.labels(symbol=symbol, type='commission', exchange='binance')._value.get()
        profit_gauge.labels(symbol=symbol, type='profit', exchange='binance').set(last_profit + profit)
        profit_gauge.labels(symbol=symbol, type='commission', exchange='binance').set(last_commission - commission)
        print(symbol, 'Binance order filled', profit, commission)
    elif bingx_fiiled_match:
        order_id = bingx_fiiled_match.group(1)
        df = asyncio.run(get_bingx_profolio_df(symbol))
        order = df[df['orderId'] == int(order_id)]
        profit = order['profit'].sum()
        commission = order['commission'].sum()
        last_profit = profit_gauge.labels(symbol=symbol, type='profit', exchange='bingx')._value.get()
        last_commission = profit_gauge.labels(symbol=symbol, type='commission', exchange='bingx')._value.get()
        profit_gauge.labels(symbol=symbol, type='profit', exchange='bingx').set(last_profit + profit)
        profit_gauge.labels(symbol=symbol, type='commission', exchange='bingx').set(last_commission + commission)
        print(symbol, 'BingX order filled', profit, commission)

def parse_log_line(line):
    pattern = re.compile(r'\[(.*?)\]\[(.*?)\]\[(.*?)\] (.*)')
    match = pattern.match(line)
    if match:
        timestamp, log_level, logger_name, message = match.groups()
        return {
            'timestamp': timestamp,
            'log_level': log_level,
            'logger_name': logger_name,
            'message': message
        }
    else:
        return None

def watch_logs(log_pattern):
    log_files = glob.glob(log_pattern)
    if not log_files:
        print("未找到符合條件的日誌文件。")
        return

    print("找到的日誌文件：", log_files)

    file_positions = {log_file: 0 for log_file in log_files}

    import time
    get_balance_time = time.time()
    try:
        while True:
            if get_balance_time + 60 < time.time():
                try:
                    binance_balance = asyncio.run(binance_exchange.get_balance())
                    bingx_balance = asyncio.run(bingx_exchange.get_balance())
                    balance_gauge.labels(symbol='USDT', exchange='binance').set(binance_balance)
                    balance_gauge.labels(symbol='USDT', exchange='bingx').set(bingx_balance)
                    get_balance_time = time.time()
                except Exception as e:
                    print("獲取餘額失敗。")

            for log_file in log_files:
                if os.path.exists(log_file) and os.path.getsize(log_file) > file_positions[log_file]:
                    with open(log_file, 'r') as f:
                        f.seek(file_positions[log_file])
                        for line in f:
                            parsed_log = parse_log_line(line)
                            if parsed_log:
                                handler(parsed_log)
                        file_positions[log_file] = f.tell()
            heartbeat_gauge.set_to_current_time()
            try:
                push_to_gateway(os.getenv('PUSHGATEWAY_URL'), os.getenv('PUSHGATEWAY_JOB'), registry)
            except Exception as e:
                print("推送失敗。")
            time.sleep(1)

    except KeyboardInterrupt:
        print("觀察已結束。")

if __name__ == "__main__":
    log_pattern = "*.log.txt"
    start_http_server(8000)
    watch_logs(log_pattern)

