from dataclasses import dataclass
import pandas as pd
import numpy as np 
import sys
import os
import datetime as dt
from datetime import datetime, timedelta
import calendar

sys.path.insert(1, os.path.dirname(__file__) + '/../../..')

from tda_symbol import ETF, EQUITY,TEST_EQUITY
from historical_price import *

# working fine in period
aapl_df = get_tda_historical_using_period(symbol = "AAPL")
print(aapl_df)
print('-------')

"""
epoch example
epoch = dt.datetime(2022, 9, 1, 0, 0, 0).timestamp()
print(epoch)
print(int(epoch))

testing area

using million second
sep_2022 = int(dt.datetime(2022, 9, 1, 0, 0, 0).timestamp())
aug_2022 = int(dt.datetime(2022, 8, 1, 0, 0, 0).timestamp())
print(sep_2022)
print(aug_2022)
tsla_df = get_tda_historical_px_kwargs(symbol="TSLA", startDate="aug_2022", endDate="sep_2022")
print(tsla_df)
print('-------')

def anchor_list(current_update):

    delta = 3850000000 # the million second delta
    start = 946656000000 # 2000-01-01 00:00:00
    anchor = []
    times = (current_update-start)/delta
    for i in range(int(times+1)):
        anchor.append(current_update)
        current_update -= delta
    return anchor

timelist = anchor_list(1663862400000) 

def test(symbol,start,end):
    # using million second test 
    url = f'https://api.tdameritrade.com/v1/marketdata/{symbol}/pricehistory'
    params={
        'apikey': key,
        # 'periodType': 'day',
        'frequencyType': 'minute',
        'frequency': '1',
        'needExtendedHoursData': 'true',
        'startDate':start,
        'endDate':end,
        }
    data = requests.get(url=url, params=params).json()
    px = []
    for i in range(len(data['candles'])):
        px_dict = dict(
            datetime = data['candles'][i]['datetime'],
            open = data['candles'][i]['open'],
            high = data['candles'][i]['high'],
            low = data['candles'][i]['low'],
            close = data['candles'][i]['close'],
            volume = data['candles'][i]['volume'],
        )
        px.append(px_dict)
    print(px)
    px = pd.DataFrame(px)
    px['datetime'] = pd.to_datetime(px['datetime'], unit='ms')

    return px

a = test('AAPL',str(timelist[1]),str(timelist[0]))    #1659974400000 #1657296000000
print(a)
a = []
for j in range(len(timelist)):
     b =  test('AAPL',str(timelist[j+1]),str(timelist[j]))  
     print(b)
print("----------")


# uodate monthly 

print(anchor_list(sep_2022))

test_anchor = [949740400000, 949740400000]
# print(get_tda_historical_px_ms(symbol="AAPL"))

ms_list = anchor_list(sep_2022)

# print(test())

# a = anchor_list(sep_2022)
# print(a)
# test_anchor = [1658140400000, 1654290400000]

# for asset in TEST_EQUITY:
#     for i in range(len(test_anchor)):

#         print(asset)
#         print(i)
#         print(test_anchor[i])
#         print(test_anchor[i+1])

#         data = test(asset,  str(ms_list[i+1]), str(ms_list[i]))
#         print(data)


# print(get_tda_historical_px_kwargs(symbol = "AAPL"))





for i in range(len(test_anchor)):
    for asset in TEST_EQUITY:
        print(asset)
        print(i)
        print(test_anchor[i])
        print(test_anchor[i+1])

        data = kwargs_test(symbol=asset, startDate=str(ms_list[i]), endDate=str(ms_list[i]))
        print(data)

"""
