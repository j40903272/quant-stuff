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
    binance_balance = await binance_exchange.get_balance()
    bingx_balance = await bingx_exchange.get_balance()
    print("Binance balance: ", binance_balance)
    print("Bingx balance: ", bingx_balance)
    nowtime = datetime.datetime.now()
    print(nowtime, "Total balance: ", binance_balance + bingx_balance)

if __name__ == "__main__":
    asyncio.run(main())