import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import configparser
import argparse
from datetime import datetime

warnings.filterwarnings("ignore")
user_home_path = str(Path.home())
sys.path.append(f"{user_home_path}/backtest_system")

config = configparser.ConfigParser()
config.read(f'{user_home_path}/backtest_system/config/data_loader_config.ini')
OANDA_ACCOUNT = config['OANDA']['ACCOUNT_NUM']
OANDA_API = config['OANDA']['API_KEY']
OANDA_URL = config['OANDA']['OANDA_URL']
OATH = {"Authorization": f"Bearer {OANDA_API}"}

session = requests.Session()

today_date = datetime.today().strftime('%Y-%m-%d')

instrument = "EUR_USD"
count = 10
granularity = "M1"
start = "2020-01-01"
end = today_date

start = pd.to_datetime(start)
end = pd.to_datetime(end)

# for bar
url = f"{OANDA_URL}/instruments/{instrument}/candles"
params = dict(count=count, granularity=granularity, price="MBA")
response = session.get(url, params=params, headers=OATH)

data = response.json()
candles = data["candles"]

mid_bar_data = []
ask_bar_data = []
bid_bar_data = []
for item in candles:
    mid = dict(
        time=item["time"],
        open=item["mid"]["o"],
        high=item["mid"]["h"],
        low=item["mid"]["l"],
        close=item["mid"]["c"],
        volume=item["volume"],
    )
    ask = dict(
        time=item["time"],
        open=item["mid"]["o"],
        high=item["mid"]["h"],
        low=item["mid"]["l"],
        close=item["mid"]["c"],
        volume=item["volume"],
    )
    bid = dict(
        time=item["time"],
        open=item["bid"]["o"],
        high=item["bid"]["h"],
        low=item["bid"]["l"],
        close=item["bid"]["c"],
        volume=item["volume"],
    )
    mid_bar_data.append(mid)
    ask_bar_data.append(ask)
    bid_bar_data.append(bid)
mid_bar_data = pd.DataFrame(ask_bar_data)
ask_bar_data = pd.DataFrame(ask_bar_data)
bid_bar_data = pd.DataFrame(ask_bar_data)

print(bid_bar_data)


