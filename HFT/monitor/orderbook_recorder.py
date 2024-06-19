import argparse
import asyncio
import threading
from okex_exchange import OKEX
from Exchange.binance_exchange import BINANCE
from bitget_exchange import BITGET
import csv
from datetime import date
import os
import pandas as pd
    
def handler(exchange, msg):
    today = date.today()
    
    # Binance depth response:
    # {
    #   "e": "depthUpdate", // Event type
    #   "E": 123456789,     // Event time
    #   "T": 123456788,     // Transaction time 
    #   "s": "BTCUSDT",     // Symbol
    #   "U": 157,           // First update ID in event
    #   "u": 160,           // Final update ID in event
    #   "pu": 149,          // Final update Id in last stream(ie `u` in last stream)
    #   "b": [              // Bids to be updated
    #     [
    #       "0.0024",       // Price level to be updated
    #       "10"            // Quantity
    #     ]
    #   ],
    #   "a": [              // Asks to be updated
    #     [
    #       "0.0026",       // Price level to be updated
    #       "100"          // Quantity
    #     ]
    #   ]
    # }
    
    with open(msg, encoding='utf-8') as inputfile:
        df = pd.read_json(inputfile)
        
        symbol = df["s"]
        
        filePath = f"./depth/{exchange}/{today}_{symbol}.csv"
        
        df.to_csv('csvfile.csv', encoding='utf-8', index=False)

        if os.path.getsize(filePath) == 0: # file not exist
            with open(filePath, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['t', 'b', 'a']) # csv header
        
        with open(filePath, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='symbol')
    parser.add_argument('-symbol', help="symbol", required=True)
    args = parser.parse_args()

    symbol = args.symbol

    binance = BINANCE(symbol)
    okex = OKEX(symbol)

    threading.Thread(target=binance.start_ws, args=(symbol, "depth", handler)).start()
    
    threading.Thread(target=okex.start_ws, args=("books")).start()