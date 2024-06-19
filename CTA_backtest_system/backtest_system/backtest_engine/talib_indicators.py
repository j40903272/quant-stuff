import os
import pandas as pd
import talib  as ta
import numpy as np
from backtesting import Backtest,Strategy

from pathlib import Path
import sys

sys.path.insert(1, os.path.dirname(__file__) + '/../..')
user_home_path = str(Path.home())

# using TA-Lib create normal and self-design indictors
# http://mrjbq7.github.io/ta-lib/doc_index.html

def avg_price(open, high, low, close):
    # AVGPRICE - Average Price
    avg_price = ta.AVGPRICE(open, high, low, close)
    return avg_price

def med_price(high, low, close):
    # MEDPRICE - Median Price
    med_price = ta.MEDPRICE( high, low, close)
    return med_price

def sma_cal(price, period: int):
    # SMA - Simple Moving Average
    sma = ta.SMA(price, timeperiod = period)
    return sma

def ema_cal(price, period: int):
    ema = ta.EMA(price, timeperiod = period)
    return ema

def kaufman_ma_cal(price, period: int):
    # KAMA - Kaufman Adaptive Moving Average
    kama = ta.KAMA(price, period=30)
    return kama

def tema_cal(price, period: int):
    # TEMA - Triple Exponential Moving Average
    tema = ta.TEMA(price, timeperiod=period)
    return tema

def bband_cal(price, period: int, width: float):
    upperband, middleband, lowerband = ta.BBANDS(price, timeperiod = period, nbdevup=width, nbdevdn=width, matype=0)
    return upperband, middleband, lowerband

def macd_cal(price, fastperiod:int, slowperiod:int, signalperiod:int):
    # MACD - Moving Average Convergence/Divergence
    macd, macdsignal, macdhist = ta.MACD(price, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    return macd, macdsignal, macdhist

def true_range_cal(high, low, close):
    # TRANGE - True Range
    tr = ta.TRANGE(high, low, close)
    return tr

def atr_cal(high, low, close, period:int):
    # ATR - Average True Range
    atr = ta.ATR(high, low, close, timeperiod=period)
    return atr

def stddev_cal(price, period:int):
    # STDDEV - Standard Deviation
    stddev = ta.STDDEV(price, timeperiod=period, nbdev=1)
    return stddev

def variance_cal(price, period: int):
    # VAR - Variance
    var = ta.VAR(price, timeperiod=period, nbdev=1)
    return var

def rsi_cal(price, period: int):
    # RSI - Relative Strength Index
    rsi = ta.RSI(price, timeperiod=period)
    return rsi

def stoch_cal(high, low, close, fastk_period: int, slowk_period: int):
    # STOCH - Stochastic
    slowk, slowd = ta.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=0, slowd_period=3, slowd_matype=0)
    return slowk, slowd

def adx_cal(high, low, close, period: int):
    # ADX - Average Directional Movement Index
    adx = ta.ADX(high, low, close, timeperiod=period)
    return adx 

def minus_di(high, low, close, period: int):
    # MINUS_DI - Minus Directional Indicator
    minus_di= ta.MINUS_DI(high, low, close, timeperiod=period)
    return minus_di

def plus_di(high, low, close, period: int):
    # PLUS_DI - Plus Directional Indicator
    plus_di= ta.PLUS_DI(high, low, close, timeperiod=period)
    return plus_di

def sar_cal(high, low):
    # SAR - Parabolic SAR
    sar = ta.SAR(high, low, acceleration=0, maximum=0)
    return sar

def aroon_cal(high, low, period:int):
    # AROON - Aroon
    aroondown, aroonup = ta.AROON(high, low, timeperiod=period)
    return aroondown, aroonup

def cci_cal(high, low, close, period: int):
    # CCI - Commodity Channel Index
    cci = ta.CCI(high, low, close, timeperiod=period)
    return cci

def cmo_cal(price, period: int):
    # CMO - Chande Momentum Oscillator
    cmo = ta.CMO(price, timeperiod=period)
    return cmo

def trix_cal(price, period: int):
    # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    trix = ta.TRIX(price, timeperiod=period)
    return trix

def willr_cal(high, low, close, period: int):
    # WILLR - Williams' %R
    willr = ta.WILLR(high, low, close, timeperiod=period)
    return willr

def linear_slope_cal(price, period: int):
    # LINEARREG_SLOPE - Linear Regression Slope
    linear_slope = ta.LINEARREG_SLOPE(price, timeperiod=period)
    return linear_slope

def roc(price, period: int):
    # ROC - Rate of change : ((price/prevPrice)-1)*100
    ros = ta.ROC(price, timeperiod=period)
    return ros

def rocp(price, period: int):
    # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
    rocp = ta.ROCP(price, timeperiod=period)
    return rocp