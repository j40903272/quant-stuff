import pandas as pd
import numpy as np

def preprocess_data(file_path):
    # Read the CSV file using pyarrow engine
    df = pd.read_csv(file_path, engine="pyarrow")

    # Rename columns
    df.columns = ['timestamp', 'bid', 'ask']

    # Convert timestamp to integer
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Set timestamp as the index
    df.set_index('timestamp', inplace=True)

    return df

def split_and_save_dataframe(df, n, file_prefix='', file_format='csv'):
    """
    將 DataFrame 分割成 n 個部分，並儲存成檔案

    Parameters:
    - df: 要分割的 DataFrame
    - n: 分割的檔案數量
    - file_prefix: 儲存檔案的前綴，預設為 'part_'
    - file_format: 儲存檔案的格式，預設為 'csv'
    """

    # 使用 numpy.array_split 將 DataFrame 分割成 n 個部分
    df_parts = np.array_split(df, n)

    # 逐一儲存每個部分到檔案
    for i, part in enumerate(df_parts):
        file_name = f'{file_prefix}{i+1}.{file_format}'
        if file_format == 'csv':
            part.to_csv(file_name, index=True)
            part = part.resample('L').ffill()
            part.to_csv(f'{file_prefix}{i+1}_filled.csv')
        # 如果有其他支援的檔案格式，可以在這裡加入相應的儲存方法
            
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

# Load and Split Data    
# binance = preprocess_data('./binance.csv')
# okx = preprocess_data('./okx.csv')

# split_and_save_dataframe(binance, n=10, file_prefix='./binance/', file_format='csv')
# split_and_save_dataframe(okx, n=1, file_prefix='./okx/', file_format='csv')

# Data Align and Save
binance = pd.read_csv('./binance/1_filled.csv',engine="pyarrow")
binance['timestamp'] = (binance['timestamp'].astype(int) / 1000000).round().astype(int)
binance.set_index('timestamp', inplace=True)

okx = pd.read_csv('./okx/1_filled.csv', engine="pyarrow")
okx['timestamp'] = (okx['timestamp'].astype(int) / 1000000).round().astype(int)
okx.set_index('timestamp', inplace=True)

time_start = max(binance.index[0],okx.index[0])
time_end = min(binance.index[-1],okx.index[-1])

aligned_binance, aligned_okx = align_dataframes_by_time(binance, okx)

aligned_binance.to_csv('./aligned_binance.csv', index=True)
aligned_okx.to_csv('./aligned_okx.csv', index=True)
