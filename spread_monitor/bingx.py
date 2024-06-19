
import json
import websocket
import gzip
import io
import requests
import time

def fetch_common_usdt_symbols():
    # Fetch symbols from BingX
    bingx_response = requests.get('https://open-api.bingx.com/openApi/swap/v2/quote/contracts')
    if bingx_response.status_code == 200:
        bingx_symbols = [symbol['symbol'].replace("-", "") for symbol in bingx_response.json()['data'] if symbol['symbol'].endswith('USDT')]
    else:
        raise Exception(f"BingX Error: {bingx_response.status_code}\n{bingx_response.text}")

    # Fetch symbols from Binance
    binance_response = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr")
    if binance_response.status_code == 200:
        data = binance_response.json()
        # Sort by 24-hour trading volume * weighted average price, descending
        sorted_data = sorted(data, key=lambda x: float(x['volume']) * float(x['weightedAvgPrice']), reverse=True)
        # Extract symbols for the top 150 futures, assuming the sorting has already been done
        binance_symbols = [item['symbol'] for item in sorted_data[:49] if 'USDT' in item['symbol']]
    else:
        raise Exception(f"Binance Error: {binance_response.status_code}\n{binance_response.text}")

    return list(set(bingx_symbols) & set(binance_symbols))

URL="wss://open-api-swap.bingx.com/swap-market" 
CHANNEL= {"id":"e745cd6d-d0f6-4a70-8d5a-043e4c741b40","reqType": "sub","dataType":"BTC-USDT@depth5@500ms"}
class BingxTest(object):

    def __init__(self, symbols):
        self.url = URL 
        self.ws = None
        self.symbols = [symbol.replace('USDT', '-USDT') for symbol in symbols]

    def on_open(self, ws):
        print('WebSocket connected')
        depth = 5
        interval = 100
        for channel in [f'{symbol}@depth{depth}@{interval}ms' for symbol in self.symbols]:
            subStr = json.dumps({"id": "e745cd6d-d0f6-4a70-8d5a-043e4c741b40", "reqType": "sub", "dataType": channel})
            ws.send(subStr)
            print("Subscribed to :",subStr)
            #time.sleep(1)

    def handler(self, message):
        pass

    def on_data(self, ws, string, type, continue_flag):
        compressed_data = gzip.GzipFile(fileobj=io.BytesIO(string), mode='rb')
        decompressed_data = compressed_data.read()
        utf8_data = decompressed_data.decode('utf-8')

    def on_message(self, ws, message):
        compressed_data = gzip.GzipFile(fileobj=io.BytesIO(message), mode='rb')
        decompressed_data = compressed_data.read()
        utf8_data = decompressed_data.decode('utf-8')
        #print(utf8_data)  #this is the message you need 
        
        if utf8_data == "Ping": # this is very important , if you receive 'Ping' you need to send 'Pong' 
           ws.send("Pong")
        else:
            self.handler(json.loads(utf8_data))

    def on_error(self, ws, error):
        print('Erorr:',error)

    def on_close(self, ws, close_status_code, close_msg):
        print('The connection is closed!')
        print("close_status_code:", close_status_code)
        print("close_msg:", close_msg)

    def start(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            # on_data=self.on_data,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        self.ws.run_forever()


# if __name__ == "__main__":
#     test = Test(fetch_common_usdt_symbols())
#     print(len())
#     #test.start()
    