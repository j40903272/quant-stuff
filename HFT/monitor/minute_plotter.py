
import requests
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PLOTSAVER():
    def __init__(self):
        pass
    
    def get_bybit_kline(self, symbol, interval, from_time):
        url = f"https://api.bybit.com/public/linear/kline?symbol={symbol}&interval={interval}&from={from_time}&limit=200"
        response = requests.get(url)
        data = response.json()
        return data['result']

    def get_bybit_all_kline(self, symbol, interval, start_date, end_date):
        start_timestamp = int(start_date.timestamp())
        end_date = int(end_date.timestamp())
        all_data = []
        while start_timestamp < end_date:
            data = self.get_bybit_kline(symbol, interval, start_timestamp)
            all_data.extend(data)
            start_timestamp += 200 * 60 
            time.sleep(0.5)  # avoid rate limit
        return all_data

    def get_bitget_kline(self, symbol, start_time, end_time):
        url = f"https://api.bitget.com/api/mix/v1/market/history-candles?symbol={symbol}&granularity=1m&startTime={start_time}&endTime={end_time}&limit=200"
        response = requests.get(url)
        data = response.json()
        return data

    def get_bitget_all_kline(self,symbol,start_date, end_date):
        
        start_time = int(start_date.timestamp() * 1000) 
        end_time = int(end_date.timestamp() * 1000)
        
        all_data = []
        while start_time < end_time:
            end_time_temp = start_time + 60 * 200 * 1000
            data = self.get_bitget_kline(symbol, start_time, end_time_temp)
            df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'currency_volume'])
            df['time'] = df['time'].apply(self.drop_last_three_chars)
            # df['time'] = pd.to_datetime(df['time'])
            all_data.append(df)
            start_time = start_time + 60 * 200 * 1000
            time.sleep(0.5)  # 避免速率限制
            
        return pd.concat(all_data, ignore_index=True)

    def drop_last_three_chars(self,string):
        return string[:-3]
    
    def save_plot(self):
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(hours=24)

        bybit = self.get_bybit_all_kline('BTCUSDT', '1', start_date, end_date)
        bybit_df = pd.DataFrame(bybit)
        bybit_df['open_time'] = pd.to_datetime(bybit_df['open_time'], unit='s')
        bybit_df = bybit_df[:-1]
        bybit_df.reset_index(drop=True, inplace=True)

        bitget_df = self.get_bitget_all_kline('BTCUSDT_UMCBL',start_date,end_date)
        bitget_df = bitget_df.drop_duplicates(subset='time')
        bitget_df = bitget_df[1:]
        bitget_df.reset_index(drop=True, inplace=True)

        spread = ((bybit_df['close'] - bitget_df['close'].astype(float)) / ((bybit_df['close'] + bitget_df['close'].astype(float)) / 2)) * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(bybit_df.index, spread, label='spread')
        ax.axhline(y=0, color='red', linestyle='-', label='0')
        ax.axhline(y=0.05, color='red', linestyle='--', label='0')
        ax.axhline(y=-0.05, color='red', linestyle='--', label='0')
        fig.savefig('./data/daily_spread_1m.png',bbox_inches='tight')
