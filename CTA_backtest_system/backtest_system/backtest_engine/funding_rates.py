import configparser
import os

import pandas as pd

'''
A module to enquire funding rates
'''
class FundingRates:
    '''
    FTX funding rates only
    '''
    def __init__(self, exchange='FTX'):
        # hardcode index_col as time, regard this as a requirement for funding rate file
        self.config = configparser.ConfigParser()
        self.config.read(f'{os.path.dirname(__file__)}/../data_center/config/data_download_config.ini')
        
        if exchange == 'FTX':
            self._path ='/Users/chouwilliam/backtest_system/data_center/csv/binance/FUNDING/funding.csv'
        else:
            raise NotImplementedError
        
        self.funding_rate_symbols=None
        if os.path.isfile(self._path):
            self._rate_df = pd.read_csv(self._path, index_col='time')
            self._rate_df.index = pd.to_datetime(self._rate_df.index, format='%Y-%m-%d %H:%M:%S')
            self.funding_rate_symbols = self._rate_df.columns

    def get_rate(self, symbol):
        if 'UPERP' in symbol or '-PERP' in symbol:
            asset = symbol.split('USDT')[0]
            if self.funding_rate_symbols is not None:
                if (perp_name := f'{asset}-PERP') in self.funding_rate_symbols:
                    rates = self._rate_df[perp_name]
                    rates.name = 'funding'
                    return rates
            else:
                return None
        else:
            return None