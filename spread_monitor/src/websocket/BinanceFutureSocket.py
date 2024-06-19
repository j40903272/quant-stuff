import io
import gzip
import json
import uuid
import requests
import websocket
from .SocketBase import SocketBase

class BinanceFutureSocket(SocketBase):
    def __init__(self, channels=[]):
        super().__init__(f"wss://fstream.binance.com/ws")
        self.channels = channels

    def on_open(self, ws):
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": self.channels,
            "id": 1,
        }
        ws.send(json.dumps(subscribe_message))
