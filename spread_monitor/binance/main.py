import time
import requests
from binance import ThreadedWebsocketManager

import requests

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
        binance_symbols = [item['symbol'] for item in sorted_data[:150] if 'USDT' in item['symbol']]
    else:
        raise Exception(f"Binance Error: {binance_response.status_code}\n{binance_response.text}")

    return list(set(bingx_symbols) & set(binance_symbols))


api_key = 'WMLrJBKHvPY6rlY80sfrIU7WQGUpc8gNhUd02IGEujw0fYtIy40hV4XOqJrjHtJl'
api_secret = 'X8j7wXDggBdvaTC6goweMYfJgFUHOiIeBl3mp5wlhjklVqyEA25K1MuHrV27EDOw'

symbols = fetch_common_usdt_symbols()

def main():

    #symbol = 'BNBUSDT'

    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
    # start is required to initialise its internal loop
    twm.start()

    def handle_socket_message(msg):
        #print(f"message type: {msg['e']}")
        print(msg)

    #twm.start_kline_socket(callback=handle_socket_message, symbol=symbol)

    # multiple sockets can be started
    # twm.start_depth_socket(callback=handle_socket_message, symbol=symbol)

    # or a multiplex socket can be started like this
    # see Binance docs for stream names
    streams = [f'{symbol.lower()}@bookTicker' for symbol in symbols]
    twm.start_multiplex_socket(callback=handle_socket_message, streams=streams)
    twm.join()


if __name__ == "__main__":
   main()
    #data = 
    #print(data)