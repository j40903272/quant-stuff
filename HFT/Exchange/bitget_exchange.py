from functools import partial
import json
import websocket
import bitget.mix.order_api as order
from bitget.consts import CONTRACT_WS_URL
from bitget.ws.bitget_ws_client import BitgetWsClient, SubscribeReq

def handel_error(message):
    print("handle_error:" + message)

class BITGET:
    def __init__(self, symbol):
        self.api_key = "bg_752ad3548ee540dd10747a87f57e033a"
        self.secret_key = "0840c68d5e1bb22cf14b7cc68402f25a055d0a40982b81f93571723da49f9f73"
        self.passphrase = "Up3cj04jp6"  # Password
        self.orderApi = order.OrderApi(self.api_key, self.secret_key, self.passphrase, use_server_time=False, first=False)
        self.bitget_client = BitgetWsClient(CONTRACT_WS_URL, need_login=True) \
            .api_key(self.api_key) \
            .api_secret_key(self.secret_key) \
            .passphrase(self.passphrase) \
            .error_listener(handel_error) \
            .build()
        self.bitget_symbol = f'{symbol}USDT_UMCBL'
        self.bitget_ws_symbol = f'{symbol}USDT'

    def on_message(self, ws, message, callback):
        result = json.loads(message)
        result['exchange'] = 'bitget'
        callback(result)

    def on_error(self, ws, error):
        print("bitget websocket error: ", type(error), error)

    def on_close(self, ws, close_status_code, close_msg):
        print("### bitget websocket closed ###")

    def on_open(self, ws):
        ws.send(json.dumps({
                "op":"subscribe",
                "args":[
                    {
                        "instType": "mc",
                        "channel": self.channel,
                        "instId": self.bitget_ws_symbol
                    }
                ]
            }))
    
    def start_ws(self, channel, callback):
        self.channel = channel
        ws = websocket.WebSocketApp("wss://ws.bitget.com/mix/v1/stream",
                                    on_open=self.on_open,
                                    on_message=partial(self.on_message, callback=callback),
                                    on_error=self.on_error,
                                    on_close=self.on_close)

        ws.run_forever(reconnect=5)