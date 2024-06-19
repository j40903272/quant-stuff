import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from datetime import datetime
from alive_progress import alive_bar
import configparser

def preprocess_data(file_path):
    
    df = pd.read_csv(file_path, engine="pyarrow")

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    df = df.resample('L').ffill()
    df = df.reset_index()

    df['timestamp'] = (df['timestamp'].astype(int) / 1000000).round().astype(int)
    df.set_index('timestamp', inplace=True)

    return df

def align_dataframes_by_time(binance, okx):
    """
    對兩個 DataFrame 按時間對齊，使它們的時間範圍相同

    Parameters:
    - binance: 第一個 DataFrame
    - okx: 第二個 DataFrame

    Returns:
    - aligned_binance: 對齊後的第一個 DataFrame
    - aligned_okx: 對齊後的第二個 DataFrame
    """

    # 計算對齊的時間範圍
    time_start = max(binance.index[0], okx.index[0])
    time_end = min(binance.index[-1], okx.index[-1])

    # 使用布林索引將兩個 DataFrame 按照計算得到的時間範圍進行過濾
    aligned_binance = binance[(binance.index >= time_start) & (binance.index <= time_end)]
    aligned_okx = okx[(okx.index >= time_start) & (okx.index <= time_end)]

    return aligned_binance, aligned_okx

def get_filtered_dataframe(dataframe, column_name, min_threshold, max_threshold):

    dataframe = dataframe[(dataframe[column_name] > min_threshold) & (dataframe[column_name] < max_threshold)]

    # Sort the DataFrame based on 'bid_change_percentage' in descending order
    dataframe = dataframe.sort_values(column_name, ascending=False)
    # print(min_threshold, max_threshold, len(dataframe))

    return dataframe

def calculate_profit(binance_long_entries, binance_short_entries, okx, MIN_LATENCY, MAX_LATENCY, MIN_OFFSET, MAX_OFFSET, MIN_ACCUMULATED_PROFIT, MAX_ACCUMULATED_PROFIT):
    result = []

    for transation_latency in range(MIN_LATENCY, MAX_LATENCY):
        for offset in range(MIN_OFFSET, MAX_OFFSET):

            accumulated_profit = 0
            for i in binance_long_entries.index.tolist():
                open_time = i + transation_latency 
                close_time = open_time + offset
                accumulated_profit += okx.loc[close_time]['bid'] - okx.loc[open_time]['ask']
            if(accumulated_profit == 0):
                continue
            accumulated_profit = accumulated_profit * 100 * (1 - 0.1) # Convert to Percent, and then minus the commission fee
            if  accumulated_profit > MIN_ACCUMULATED_PROFIT and accumulated_profit < MAX_ACCUMULATED_PROFIT and accumulated_profit:
                result.append({
                    'transation_latency': transation_latency,
                    'offset': offset,
                    'accumulated_profit': accumulated_profit
                })
            
            accumulated_profit = 0
            for i in binance_short_entries.index.tolist():
                #print(i)
                open_time = i + transation_latency 
                close_time = open_time + offset
                accumulated_profit += okx.loc[open_time]['bid'] - okx.loc[close_time]['ask']
            if(accumulated_profit == 0):
                continue

            accumulated_profit = accumulated_profit * 100 * (1 - 0.1) # Convert to Percent, and then minus the commission fee
            if  accumulated_profit > MIN_ACCUMULATED_PROFIT and accumulated_profit < MAX_ACCUMULATED_PROFIT and accumulated_profit:
                result.append({
                    'transation_latency': transation_latency,
                    'offset': offset,
                    'accumulated_profit': accumulated_profit
                })

    transation_latency_values = [item['transation_latency'] for item in result]
    accumulated_profit_values = [item['accumulated_profit'] for item in result]
    offset_values = [item['offset'] for item in result]

    return transation_latency_values, offset_values, accumulated_profit_values

try:
    config = configparser.ConfigParser()
    config.read('./config.ini')

except FileNotFoundError:
    print(f"找不到檔案")
except Exception as e:
    print(f"讀取檔案時發生錯誤：{str(e)}")

DEBUG =str(config['DEFUALT']['DEBUG']) == 'True'

MIN_LATENCY = 25
MAX_LATENCY = 60

MIN_OFFSET = 15
MAX_OFFSET = 35

MAX_ACCUMULATED_PROFIT = 4
MIN_ACCUMULATED_PROFIT = -1

MIN_THRESHOLD =  float(config['DEFUALT']['threshold'])
MAX_THRESHOLD =  1 #設定為1暫時沒用

while(1):
    time = datetime.now()
    entry = True
    if not DEBUG:
        entry = time.strftime('%M') == '00' and time.strftime('%S') == '00'
    if(entry):
        directory = './symbols_Raw/binance'
        file_names = os.listdir(directory)

        with alive_bar(len(file_names)) as bar:
            for file_name in file_names:
                binance = preprocess_data(f'./symbols_Raw/binance/{file_name}')
                okx = preprocess_data(f'./symbols_Raw/okx/{file_name}')

                binance, okx = align_dataframes_by_time(binance, okx)

                binance['bid_change_percentage'] = ((binance['bid'] - binance['bid'].shift(1)) / binance['bid'].shift(1))

                binance_long_entries = get_filtered_dataframe(binance, 'bid_change_percentage', MIN_THRESHOLD, MAX_THRESHOLD)
                binance_short_entries = get_filtered_dataframe(binance, 'bid_change_percentage', -MAX_THRESHOLD, -MIN_THRESHOLD)

                transation_latency_values, offset_values, accumulated_profit_values = calculate_profit(binance_long_entries, binance_short_entries, okx, MIN_LATENCY, MAX_LATENCY, MIN_OFFSET, MAX_OFFSET, MIN_ACCUMULATED_PROFIT, MAX_ACCUMULATED_PROFIT)
                
                symbol = file_name.replace('.csv','')
                start = datetime.fromtimestamp(binance.index[0]/1000).strftime('%Y-%m-%d_%H-%M')
                end  = datetime.fromtimestamp(binance.index[-1]/1000).strftime('%Y-%m-%d_%H-%M')
                title = f'{symbol}_{start}_To_{end}_{MIN_THRESHOLD}_{MAX_THRESHOLD}'

                #print(f'{symbol} trades:{len(transation_latency_values)}')

                # Plot the data
                plt.clf()
                plt.figure(figsize=(12,12))

                # 设置背景颜色渐变
                gradient = np.linspace(1, 0, 256).reshape(-1, 1)
                gradient = np.hstack((gradient, gradient))

                plt.imshow(gradient, aspect='auto', extent=(MIN_LATENCY, MAX_LATENCY, 0, MAX_ACCUMULATED_PROFIT), cmap='Greens', alpha=0.5)
                plt.imshow(gradient, aspect='auto', extent=(MIN_LATENCY, MAX_LATENCY, 0, MIN_ACCUMULATED_PROFIT), cmap='Reds', alpha=0.5)

                #RdYlGn_r
                
                plt.scatter(transation_latency_values, accumulated_profit_values, c='blue', s=5)
                plt.title(title)
                plt.xticks(np.arange(MIN_LATENCY, MAX_LATENCY, 1))  
                plt.ylim(MIN_ACCUMULATED_PROFIT, MAX_ACCUMULATED_PROFIT)
                plt.xlabel('Transaction Latency')
                plt.ylabel('Accumulated Profit')

                position_texts = {}
                for i, text in enumerate(offset_values):
                    position = (transation_latency_values[i], accumulated_profit_values[i])
                    if position not in position_texts or text < position_texts[position]:
                        position_texts[position] = text

                texts = [plt.text(position[0], position[1], txt, size=9) for position, txt in position_texts.items()]

                adjust_text(texts)

                plt.savefig('./result/' + title + '.png', dpi = 100)

                bar()

        if(DEBUG):
            break
        #break