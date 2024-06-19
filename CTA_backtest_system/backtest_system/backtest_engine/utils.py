import numpy as np
import pandas as pd
from datetime import datetime, timedelta

LONG_POSITION = 'is_long'
SHORT_POSITION = 'is_short'

def get_timeframe_string(portfolio_df):

    timeframe = portfolio_df.index.to_series().diff()[-1]
    if timeframe == timedelta(minutes=5):
        timeframe_in_string = '5m'
    elif timeframe == timedelta(minutes=15):
        timeframe_in_string = '15m'
    elif timeframe == timedelta(minutes=30):
        timeframe_in_string = '30m'
    elif timeframe == timedelta(hours=1):
        timeframe_in_string = '1h'
    elif timeframe == timedelta(hours=2):
        timeframe_in_string = '2h'
    elif timeframe == timedelta(hours=4):
        timeframe_in_string = '4h'
    elif timeframe == timedelta(hours=8):
        timeframe_in_string = '8h'
    elif timeframe == timedelta(hours=12):
        timeframe_in_string = '12h'
    elif timeframe == timedelta(days=1):
        timeframe_in_string = '1d'
    elif timeframe == timedelta(days=2):
        timeframe_in_string = '2d'
    elif timeframe == timedelta(days=3):
        timeframe_in_string = '3d'

    return timeframe_in_string

def dict_without_keys(dict, keys):
   return {x: dict[x] for x in dict if x not in keys}

all_symbol = [
 'SNXUSDT_UPERP',
 'OMGUSDT_UPERP',
 'VETUSDT_UPERP',
 'SRMUSDT_UPERP',
 'ATOMUSDT_UPERP',
 'UNIUSDT_UPERP',
 'ICPUSDT_UPERP',
 'BATUSDT_UPERP',
 'RVNUSDT_UPERP',
 'HBARUSDT_UPERP',
 'SFPUSDT_UPERP',
 'NKNUSDT_UPERP',
 'HNTUSDT_UPERP', 
 'RUNEUSDT_UPERP',
 'CELRUSDT_UPERP',
 'SKLUSDT_UPERP',
 'API3USDT_UPERP',
 'FTTUSDT_UPERP',
 'BTCSTUSDT_UPERP',
 'TOMOUSDT_UPERP',
 'STMXUSDT_UPERP',
 'SXPUSDT_UPERP',
 'OGNUSDT_UPERP',
 'ANKRUSDT_UPERP', 
 'DUSKUSDT_UPERP',
 'RAYUSDT_UPERP',
 'SOLUSDT_UPERP',
 'GTCUSDT_UPERP',
 '1000SHIBUSDT_UPERP',
 'LRCUSDT_UPERP',
 'COMPUSDT_UPERP', 
 'SUSHIUSDT_UPERP',
 'HOTUSDT_UPERP', 
 'ZECUSDT_UPERP', 
 'YFIUSDT_UPERP', 
 'AVAXUSDT_UPERP', 
 'BAKEUSDT_UPERP',
 'KAVAUSDT_UPERP', 
 'KLAYUSDT_UPERP', 
 'QTUMUSDT_UPERP', 
 'LINAUSDT_UPERP', 
 'ALPHAUSDT_UPERP',
 'AAVEUSDT_UPERP', 
 'CTKUSDT_UPERP', 
 'IOSTUSDT_UPERP',
 'XMRUSDT_UPERP',
 'ETHBUSD_UPERP', 
 'MASKUSDT_UPERP',
 'ENJUSDT_UPERP', 
 'WOOUSDT_UPERP',
 'IOTAUSDT_UPERP',
 'YFIIUSDT_UPERP',
 'BTSUSDT_UPERP',
 'XRPBUSD_UPERP', 
 'LTCUSDT_UPERP', 
 'ANCUSDT_UPERP', 
 'SCUSDT_UPERP',
 'EGLDUSDT_UPERP', 
 'LINKUSDT_UPERP',
 'IMXUSDT_UPERP', 
 'CTSIUSDT_UPERP',
 'BANDUSDT_UPERP',
 'OCEANUSDT_UPERP', 
 'ADAUSDT_UPERP', 
 'SANDUSDT_UPERP', 
 'AUDIOUSDT_UPERP',
 'BNXUSDT_UPERP', 
 '1000BTTCUSDT_UPERP',
 'GALAUSDT_UPERP',
 'CHZUSDT_UPERP', 
 'MANAUSDT_UPERP',
 'ROSEUSDT_UPERP',
 'DOGEBUSD_UPERP', 
 'XLMUSDT_UPERP',
 'FILUSDT_UPERP',
 'RLCUSDT_UPERP',
 'SOLBUSD_UPERP', 
 'FLOWUSDT_UPERP', 
 'C98USDT_UPERP',
 'AXSUSDT_UPERP',
 'BCHUSDT_UPERP', 
 'ZENUSDT_UPERP',
 'XEMUSDT_UPERP', 
 'THETAUSDT_UPERP',
 'ONTUSDT_UPERP',
 'GRTUSDT_UPERP',
 'DGBUSDT_UPERP',
 'TLMUSDT_UPERP', 
 'ALGOUSDT_UPERP',
 'AVAXBUSD_UPERP',
 'MTLUSDT_UPERP', 
 'DEFIUSDT_UPERP',
 'CELOUSDT_UPERP', 
 'GMTUSDT_UPERP', 
 'FTMUSDT_UPERP',
 'ARPAUSDT_UPERP',
 'STORJUSDT_UPERP', 
 'NEARUSDT_UPERP',
 'JASMYUSDT_UPERP', 
 '1000XECUSDT_UPERP',
 'WAVESUSDT_UPERP', 
 'BLZUSDT_UPERP', 
 'CHRUSDT_UPERP',
 'CRVUSDT_UPERP',
 'TRBUSDT_UPERP', 
 'FTTBUSD_UPERP',
 'ANTUSDT_UPERP', 
 'BTCDOMUSDT_UPERP', 
 'BNBBUSD_UPERP', 
 'LPTUSDT_UPERP', 
 'GALUSDT_UPERP', 
 'AKROUSDT_UPERP',
 'ICXUSDT_UPERP', 
 'BELUSDT_UPERP', 
 'ATAUSDT_UPERP', 
 'DENTUSDT_UPERP', 
 'MATICUSDT_UPERP',
 'BTCBUSD_UPERP', 
 'RENUSDT_UPERP',
 'PEOPLEUSDT_UPERP',
 'ADABUSD_UPERP', 
 'APEUSDT_UPERP', 
 'DASHUSDT_UPERP',
 'CVCUSDT_UPERP',
 'TRXUSDT_UPERP', 
 'XTZUSDT_UPERP',
 'EOSUSDT_UPERP', 
 'DARUSDT_UPERP',
 'COTIUSDT_UPERP', 
 'DOTUSDT_UPERP', 
 'FLMUSDT_UPERP', 
 'DYDXUSDT_UPERP',
 'RSRUSDT_UPERP', 
 'DOGEUSDT_UPERP', 
 'DODOUSDT_UPERP', 
 'IOTXUSDT_UPERP', 
 'UNFIUSDT_UPERP', 
 'LITUSDT_UPERP', 
 'MKRUSDT_UPERP', 
 'ZRXUSDT_UPERP', 
 'REEFUSDT_UPERP',
 'ZILUSDT_UPERP', 
 '1INCHUSDT_UPERP',
 'BALUSDT_UPERP', 
 'KSMUSDT_UPERP', 
 'ALICEUSDT_UPERP',
 'ENSUSDT_UPERP', 
 'ETCUSDT_UPERP', 
 'NEOUSDT_UPERP',
 'ARUSDT_UPERP', 
 'ONEUSDT_UPERP', 
 'KNCUSDT_UPERP', 
 'XRPUSDT_UPERP']

target_symbol = [
    'BTCUSDT_UPERP', 
    'ETHUSDT_UPERP',
    'BNBUSDT_UPERP',
    'LINKUSDT_UPERP',
    'SOLUSDT_UPERP',
    'DOTUSDT_UPERP',
    'AAVEUSDT_UPERP',
    'ADAUSDT_UPERP',
    'UNIUSDT_UPERP',
    'NEARUSDT_UPERP',
    'TRXUSDT_UPERP',
    'MATICUSDT_UPERP',
    'MKRUSDT_UPERP', 
    'ONEUSDT_UPERP', 
    'FTMUSDT_UPERP',
    'MANAUSDT_UPERP',
    'ATOMUSDT_UPERP',
    'AVAXUSDT_UPERP',
    'SNXUSDT_UPERP',
]

major_symbol = [
    'BTCUSDT_UPERP', 
    'ETHUSDT_UPERP',
    'BNBUSDT_UPERP',
    # 'FTTUSDT_UPERP',
    'LINKUSDT_UPERP',
    'ADAUSDT_UPERP',
    'TRXUSDT_UPERP',
    'MATICUSDT_UPERP',
    'ONEUSDT_UPERP', 
    'FTMUSDT_UPERP',
    'SNXUSDT_UPERP',
]

diverse_1 = [
    'LINKUSDT_UPERP',
    'SOLUSDT_UPERP',
    'AAVEUSDT_UPERP',
    'ADAUSDT_UPERP',
]

diverse_2 = [
    'MANAUSDT_UPERP',
    'ATOMUSDT_UPERP',
    'DOTUSDT_UPERP',
    'TRXUSDT_UPERP',
]


just_btc = [
    'BTCUSDT_UPERP',
    ]


just_eth = [
    'ETHUSDT_UPERP',
    ]

just_bnb = [
    'BNBUSDT_UPERP',
]

# make some minor symbol into desperate

just_sol = [
    'SOLUSDT_UPERP',
]

just_link = [
    'LINKUSDT_UPERP',
]

just_aave = [
    'AAVEUSDT_UPERP',
]

potential = [
    'SNXUSDT_UPERP',
    'ETCUSDT_UPERP',
    'ETCUSDT_UPERP',
    'RUNEUSDT_UPERP'
]

just_dot = [
    'DOTUSDT_UPERP',
]