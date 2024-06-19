from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.append(os.getcwd())
import asyncio
import datetime
from pluto.exchange import BinanceExchange, BingxExchange

async def main():
    binance_exchange = BinanceExchange(os.getenv('BINANCE_APIKEY'), os.getenv('BINANCE_SECRET'))
    bingx_exchange = BingxExchange(os.getenv('BINGX_APIKEY'), os.getenv('BINGX_SECRET'))
    for symbol in ['REN-USDT', 'LTC-USDT', 'ARB-USDT', 'LDO-USDT', 'SUI-USDT', 'XRP-USDT', 'NEO-USDT']:
        binance_position = await binance_exchange.fetch_position(symbol.replace("-", ""))
        for position in binance_position:
            if float(position.get('positionAmt')) > 0:
                print("Binance position: ", position.get('symbol'), position.get('positionAmt'), position.get('positionSide'))
                result = await binance_exchange.place_order(symbol.replace("-", ""), 'MARKET', 'LONG', 'SELL', 0, float(position.get('positionAmt')))
                print("Binance order: ", result)

        bingx_position = await bingx_exchange.fetch_position(symbol)
        for position in bingx_position:
            if float(position.get('positionAmt')) > 0:
                print("Bingx position: ", position.get('symbol'), position.get('positionAmt'), position.get('positionSide'))
                result = await bingx_exchange.place_order(symbol, 'MARKET', 'SHORT', 'BUY', 0, float(position.get('positionAmt')))
                print("Bingx order: ", result)
    for symbol in symbols:
        binance_orders = await binance_exchange.fetch_open_orders(symbol.replace("-", ""))
        bingx_orders = await bingx_exchange.fetch_open_orders(symbol)
        print("Binance orders: ", binance_orders)
        print("Bingx orders: ", bingx_orders)

if __name__ == "__main__":
    asyncio.run(main())
