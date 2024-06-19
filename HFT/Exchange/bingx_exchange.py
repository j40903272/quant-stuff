from functools import partial
import gzip
import io
import json
import websocket
import uuid

import time
import requests
import hmac
from hashlib import sha256
from .exchange import Exchange


class BINGX(Exchange):
    def __init__(self, api_key_ini, maker_cost, taker_cost, logger=None, symbol="ETH") -> None:
        """
        Initialize an exchange object.

        Args:
            logger: Logger object for logging.
            API_URL: API URL of the exchange.
            WS_URL: WebSocket URL of the exchange.
            APIKEY: API key for accessing the exchange's API.
            SECRETKEY: Secret key part of the API key for authentication.
            name: Name of the exchange.
            symbol: Currency pair traded on the exchange.
        """
        super().__init__(
            api_key_ini=api_key_ini,
            logger=logger,
            API_URL="https://open-api.bingx.com",
            WS_URL="wss://open-api-swap.bingx.com/swap-market",
            name="BingX",
            symbol=f"{symbol.upper()}-USDT",
            maker_cost=maker_cost,
            taker_cost=taker_cost,
        )

        self.get_asset_precision()

    def get_asset_precision(self):
        payload = {}
        path = "/openApi/swap/v2/quote/contracts"
        method = "GET"
        paramsMap = {}
        paramsStr = self.parseParam(paramsMap)
        response = self.send_request(method, path, paramsStr, payload)
        msg = json.loads(response)
        for pair_info in msg["data"]:
            if pair_info["symbol"] == self.symbol:
                self.quantityPrecision = int(pair_info["quantityPrecision"])
                self.pricePrecision = int(pair_info["pricePrecision"])
                return
        raise ValueError(f"Precision not found for symbol {self.symbol}")

    def get_listen_key(self):
        payload = {}
        path = "/openApi/user/auth/userDataStream"
        method = "POST"
        paramsMap = {}
        paramsStr = self.parseParam(paramsMap)
        return json.loads(self.send_request(method, path, paramsStr, payload))

    def get_sign(self, api_secret, payload):
        signature = hmac.new(
            api_secret.encode("utf-8"), payload.encode("utf-8"), digestmod=sha256
        ).hexdigest()
        # print("sign=" + signature)
        return signature

    def send_request(self, method, path, urlpa, payload):
        url = "%s%s?%s&signature=%s" % (
            self.API_URL,
            path,
            urlpa,
            self.get_sign(self.SECRETKEY, urlpa),
        )
        # print(url)
        headers = {
            "X-BX-APIKEY": self.APIKEY,
        }
        response = requests.request(method, url, headers=headers, data=payload)
        return response.text

    def parseParam(self, paramsMap):
        sortedKeys = sorted(paramsMap)
        paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
        return paramsStr + "&timestamp=" + str(int(time.time() * 1000))

    def on_message(self, ws, message, callback):
        compressed_data = gzip.GzipFile(fileobj=io.BytesIO(message), mode="rb")
        decompressed_data = compressed_data.read()
        msg = decompressed_data.decode("utf-8")
        # print(msg)  #this is the message you need
        if (
            msg == "Ping"
        ):  # this is very important , if you receive 'Ping' you need to send 'Pong'
            ws.send("Pong")
            return
        result = json.loads(msg)
        result["exchange"] = "bingx"
        callback(result)

    def on_user_message(self, ws, message, callback):
        compressed_data = gzip.GzipFile(fileobj=io.BytesIO(message), mode="rb")
        decompressed_data = compressed_data.read()
        msg = decompressed_data.decode("utf-8")
        # print(msg)  #this is the message you need
        if (
            msg == "Ping"
        ):  # this is very important , if you receive 'Ping' you need to send 'Pong'
            ws.send("Pong")
            # print("pong")
            return

        result = json.loads(msg)
        result["exchange"] = "bingx"
        callback(result)

    def on_open(self, ws):
        data = {
            "id": str(uuid.uuid4()),
            "reqType": "sub",
            "dataType": f"{self.symbol}@{self.stream_name}",
        }
        ws.send(json.dumps(data))
        print("Bingx ws subscribed to: ", f"{self.symbol}@{self.stream_name}")

    def start_ws(self, stream_name, callback):
        if stream_name == "user_ws":
            print(self.get_listen_key())
            listen_key = self.get_listen_key()["listenKey"]
            ws = websocket.WebSocketApp(
                url=self.WS_URL + "?listenKey=" + listen_key,
                on_message=partial(self.on_user_message, callback=callback),
                on_error=self.on_error,
                on_close=self.on_close,
            )
            ws.run_forever(reconnect=5)
            print("Bingx ws url: ", self.WS_URL)
        else:
            self.stream_name = stream_name
            ws = websocket.WebSocketApp(
                url=self.WS_URL,
                on_open=self.on_open,
                on_message=partial(self.on_message, callback=callback),
                on_error=self.on_error,
                on_close=self.on_close,
            )
            ws.run_forever(reconnect=5)

    def place_order(
        self,
        quantity,
        direction,
        positionSide,
        order_type,
        leverage=None,
        stop_loss=None,
        take_profit=None,
    ):
        payload = {}
        path = "/openApi/swap/v2/trade/order"
        method = "POST"
        paramsMap = {
            "symbol": self.symbol,
            "side": direction,
            "positionSide": positionSide,
            "type": order_type,
            "quantity": quantity,
        }
        paramsStr = self.parseParam(paramsMap)
        return self.send_request(method, path, paramsStr, payload)

    def get_balance(self):
        payload = {}
        path = '/openApi/swap/v2/user/balance'
        method = "GET"
        paramsMap = {
            "recvWindow": 0
        }
        paramsStr = self.parseParam(paramsMap)
        return self.send_request(method, path, paramsStr, payload)
        
    def switch_leverage(self, leverage, side):
        payload = {}
        path = '/openApi/swap/v2/trade/leverage'
        method = "POST"
        paramsMap = {
            "symbol": self.symbol,
            "side": side,
            "leverage": leverage,
            "recvWindow": 0
        }
        paramsStr = self.parseParam(paramsMap)
        return self.send_request(method, path, paramsStr, payload)

    def get_all_position(self):
        payload = {}
        path = '/openApi/swap/v2/user/positions'
        method = "GET"
        paramsMap = {
            "recvWindow": 0
        }
        paramsStr = self.parseParam(paramsMap)
        return self.send_request(method, path, paramsStr, payload)


if __name__ == "__main__":
    bingx = BINGX(logger=None, symbol="TRB")
    # response = bingx.place_order(10, 'BUY', 'LONG', 'MARKET')
    # print(response)
