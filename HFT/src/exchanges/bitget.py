#%%
from logging import Logger
from ..sdks import *
import time
import pandas as pd
from datetime import datetime
import random
from decimal import *
import warnings
import json
import websocket
import requests
from ..sdks.bitget.ws.bitget_ws_client import *


#%%


CONTRACT_WS_URL = 'wss://ws.bitget.com/mix/v1/stream'

class Bitget():
    def __init__(self):
        self.api_key = "bg_e9399acd9ff8486bcc198353ce4db784"
        self.secret_key = "c103d17ecf5115cac2ae59788b903d482194cd090e090841cb343ff85e5301b1"
        self.passphrase = "john611030"

        self.rest = (f"https://api.bitget.com/api/mix/v1/market/depth?symbol=BTCUSDT_UMCBL&limit=15")
        
        self.ws_perp = BitgetWsClient(CONTRACT_WS_URL, need_login=True) \
            .api_key(self.api_key) \
            .api_secret_key(self.secret_key) \
            .passphrase(self.passphrase) \
            .error_listener(self.handel_error) \
            .build()
        
        self._orderbooks={}
        self._orderbooks_timestamps = {} 
        
        self.log_path = r"log"
        # self.logger = Logger(logger_folder=self.log_path,
                            #  file_name='{}'.format(self.name))
    def handel_error(message,code):

        print("handle_error:" + message, "error_code:" + code)
        
    
    def _handle_orderbooks(self, msg):# similar to unified.__handle_orderbooks or on_message()
        
        msg_dict = json.loads(msg)
        
        if msg_dict == {}:
            pass
        else:
            symbol = msg_dict['arg']['instId']
            self._orderbooks['timestamp'] = int(msg_dict['data'][0]['ts'])
            if msg_dict['action'] == 'snapshot':
                # print(msg_dict)
                self._orderbooks[symbol] = {
                    'bids': [[float(elem[0]), float(elem[1])] for elem in msg_dict['data'][0]['bids']],
                    'asks': [[float(elem[0]), float(elem[1])] for elem in msg_dict['data'][0]['asks']],
                }
      
    def get_orderbook(self):
        channel = [SubscribeReq("mc", "books15", 'BTCUSDT')]

        self.ws_perp.subscribe(channel, self._handle_orderbooks)
        if self._orderbooks == {}: 
            print('using restful bitget')
            try:
                rsp = requests.get(self.rest)
                msg_dict = rsp.json()
            except:
                time.sleep(1)
                rsp = requests.get(self.rest)
                msg_dict = rsp.json()

            symbol = 'BTCUSD'
            self._orderbooks['timestamp'] = int(msg_dict['data']['timestamp'])
            self._orderbooks[symbol] = {
                'bids': [[float(elem[0]), float(elem[1])] for elem in msg_dict['data']['bids']],
                'asks': [[float(elem[0]), float(elem[1])] for elem in msg_dict['data']['asks']],
            }
        return self._orderbooks
    
    def exist(self):
        self.ws_perp
    

        
    