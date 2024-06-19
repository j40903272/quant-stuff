import io
import re
import rel
import gzip
import json
import uuid
import time
import requests
import websocket
import traceback

class SocketBase(object):
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.close_reason = ""
        self.middlewares = []

    def handler(self, message):
        pass

    def on_open(self, ws):
        pass

    def on_message(self, ws, message):
        try:
            for middleware in self.middlewares:
                message = middleware(message)
            self.handler(message)
        except Exception as e:
            print("Error", e)
            traceback.print_exc()

    def on_error(self, ws, error):
        print("on_error", error)

    def on_close(self, ws, close_status_code, close_msg):
        self.close_reason = close_msg if close_msg else "Unknown"

    def run(self):
        self.ws = websocket.WebSocketApp(
            self.url, None, self.on_open, self.on_message, self.on_error, self.on_close
        )

        # This code is referenced from
        # https://github.com/websocket-client/websocket-client?tab=readme-ov-file#long-lived-connection
        self.ws.run_forever(reconnect=5)

    def start(self, auto_reconnect=True):
        while True:
            try:
                self.run()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print("Error", e)
            if not auto_reconnect or self.close_reason == "Unknown":
                break


BINGX_SWAP_URL = "wss://open-api-swap.bingx.com/swap-market"


class BingX(SocketBase):
    def __init__(self, channels=[]):
        super().__init__(BINGX_SWAP_URL)
        self.channels = channels

    def on_open(self, ws):
        id = str(uuid.uuid4())
        # subscribe all channels
        for channel in self.channels:
            ws.send(json.dumps({"id": id, "reqType": "sub", "dataType": channel}))

    def on_message(self, ws, message):
        compressed_data = gzip.GzipFile(fileobj=io.BytesIO(message), mode="rb")
        decompressed_data = compressed_data.read()
        utf8_data = decompressed_data.decode("utf-8")

        # ping pong, keep alive
        if utf8_data == "Ping":
            return ws.send("Pong")

        return super().on_message(ws, utf8_data)

class BingxOrderbookWebsocket(BingX):
    def __init__(self, symbol, depth=100):
        super().__init__([symbol + "@depth" + str(depth)])
        self.middlewares = [self.process_middleware]

    def process_middleware(self, message):
        j = json.loads(message)
        if j.get('data') is None or j['data'].get('bids') is None or j['data'].get('asks') is None:
            return None
        result = {}
        result['t'] = time.time_ns()
        result['bids'] = j['data']['bids']
        result['asks'] = j['data']['asks']
        return result


BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
BINANCE_FUTURE_WS_URL = "wss://fstream.binance.com/ws"


class Binance(SocketBase):
    def __init__(self, channels=[]):
        super().__init__(BINANCE_WS_URL)
        self.channels = channels
        

    def on_open(self, ws):
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": self.channels,
            "id": 1,
        }
        ws.send(json.dumps(subscribe_message))

    def on_message(self, ws, message):
        return super().on_message(ws, message)

class BinanceOrderbookSocketBase(SocketBase):
    def __init__(self, channels=[]):
        super().__init__(BINANCE_WS_URL)
        self.channels = channels

    def on_open(self, ws):
        # reset orderbook if reconnected
        self.orderbooks = {}
        self.cache = {}
        self.isRequesting = {}

        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": self.channels,
            "id": 1,
        }
        ws.send(json.dumps(subscribe_message))

    def on_message(self, ws, message):
        j = json.loads(message)
        s = j['s']
        
        if self.cache.get(s) is None:
            self.cache[s] = []
        self.cache[s].append(j)

        if self.orderbooks.get(s) is None:
            # if orderbook is not ready
            # get snapshot
            if self.isRequesting.get(s) is None:
                self.isRequesting[s] = True
                url = f"https://api.binance.com/api/v3/depth?symbol={s}&limit=100" # default is 1000
                r = requests.get(url)
                snapshot = r.json()
                self.orderbooks[s] = {"asks": {}, "bids": {}, "lastUpdateId": snapshot['lastUpdateId'], "first": True}
                for (price, quantity) in snapshot['bids']:
                    self.orderbooks[s]['bids'][price] = quantity
                for (price, quantity) in snapshot['asks']:
                    self.orderbooks[s]['asks'][price] = quantity
        else:
            while len(self.cache[s]) > 0:
                j = self.cache[s].pop(0)
                if j['u'] <= self.orderbooks[s]['lastUpdateId']:
                    continue
                if self.orderbooks[s]['first']:
                    self.orderbooks[s]['first'] = False
                else:
                    assert j['U'] == self.orderbooks[s]['lastUpdateId'] + 1
                self.orderbooks[s]['lastUpdateId'] = j['u']
                for (price, quantity) in j['b']:
                    self.orderbooks[s]['bids'][price] = quantity
                for (price, quantity) in j['a']:
                    self.orderbooks[s]['asks'][price] = quantity

        result = {}
        result['s'] = s
        result['bids'] = [list(x) for x in self.orderbooks[s]['bids'].items()]
        result['asks'] = [list(x) for x in self.orderbooks[s]['asks'].items()]
        return super().on_message(ws, json.dumps(result))

class BinanceOrderbookSocket(Binance):
    def __init__(self, symbol):
        # symbol should be the string match r"^[A-Z]+-[A-Z]+$"
        if not re.match(r"^[A-Z]+-[A-Z]+$", symbol):
            raise Exception("Invalid symbol")

        channel_name = symbol.replace("-", "").lower() + "@depth20@100ms"

        super().__init__([channel_name])
        self.symbol = symbol
        self.middlewares = [self.process_middleware]

    def process_middleware(self, message):
        j = json.loads(message)
        if j.get('bids') is None or j.get('asks') is None:
            return None
        result = {}
        result['t'] = time.time_ns()
        result['bids'] = j['bids']
        result['asks'] = j['asks']
        return result

class BinanceUserSocket(SocketBase):
    def __init__(self, apikey, serect):
        from binance.client import Client
        import sched
        import threading
        client = Client(apikey, serect)
        listen_key = client.futures_stream_get_listen_key()
        # every 30 minutes renew listenkey
        s = sched.scheduler(time.time, time.sleep)
        def func():
            client.futures_stream_keepalive(listen_key)
            s.enter(30 * 60, 1, func)
        s.enter(30 * 60, 1, func)
        threading.Thread(target=s.run).start()
        url = f"wss://fstream.binance.com/ws/{listen_key}"
        super().__init__(url)
        self.middlewares = [self.process_middleware]

    def process_middleware(self, message):
        j = json.loads(message)
        return j

class BinanceFutureBooktickerSocket(SocketBase):
    def __init__(self, symbol):
        # symbol should be the string match r"^[A-Z]+-[A-Z]+$"
        if not re.match(r"^[A-Z]+-[A-Z]+$", symbol):
            raise Exception("Invalid symbol")

        channel_name = symbol.replace("-", "").lower() + "@bookTicker"
        url = f"{BINANCE_FUTURE_WS_URL}/{channel_name}"
        super().__init__(url)
        self.symbol = symbol
        self.middlewares = [self.process_middleware]

    def process_middleware(self, message):
        j = json.loads(message)
        b = j.get('b')
        a = j.get('a')
        B = j.get('B')
        A = j.get('A')
        if b is None or a is None or B is None or A is None:
            return None
        result = {}
        result['t'] = time.time_ns()
        result['bids'] = [[b, B]]
        result['asks'] = [[a, A]]
        return result
