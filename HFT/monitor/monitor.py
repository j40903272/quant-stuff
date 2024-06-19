import argparse
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim
matplotlib.use("QtAgg") # ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
import threading
import time
import websocket
import json
import numpy as np
from collections import deque
from Exchange.binance_exchange import BINANCE
from bitget_exchange import BITGET
from okex_exchange import OKEX

bitget_asks = 0
bitget_bids = 0
bnb_asks = 0
bnb_bids = 0

max_size = 5000
bitget_ask_price_for_plot = deque(np.full([max_size], np.NaN))
bitget_bid_price_for_plot = deque(np.full([max_size], np.NaN))
bnb_ask_price_for_plot = deque(np.full([max_size], np.NaN))
bnb_bid_price_for_plot = deque(np.full([max_size], np.NaN))
arbitrage_opportunity = deque(np.full([max_size], np.nan))

mutex = threading.Lock()
i = 0
def store_value(msg):
    global i
    # print(msg)
    if 'exchange' in msg and msg['exchange'] == 'binance': # binance
        if 'e' in msg and msg['e'] == 'bookTicker':
            globals()['bnb_asks'] = float(msg["b"])
            globals()['bnb_bids'] = float(msg["a"])
    elif 'exchange' in msg and msg['exchange'] == 'bitget': # bitget
        if 'action' in msg and msg['action'] == 'snapshot' and msg['arg']['channel'] == 'books1':
            globals()['bitget_asks'] = float(msg['data'][0]['asks'][0][0])
            globals()['bitget_bids'] = float(msg['data'][0]['bids'][0][0])
    else:
        print("msg:", msg)
    
    i += 1

    mutex.acquire()

    bitget_ask_price_for_plot.append(bitget_asks)
    bitget_bid_price_for_plot.append(bitget_bids)
    bnb_ask_price_for_plot.append(bnb_asks)
    bnb_bid_price_for_plot.append(bnb_bids)
    # arbitrage_opportunity.append(bnb_asks if bitget_bids > bnb_asks or bnb_bids > bitget_asks else np.nan)

    bitget_ask_price_for_plot.popleft()
    bitget_bid_price_for_plot.popleft()
    bnb_ask_price_for_plot.popleft()
    bnb_bid_price_for_plot.popleft()
    # arbitrage_opportunity.popleft()

    mutex.release()

    # print(bnb_bid_price_for_plot)
    # print(bnb_ask_price_for_plot)
    # print(bitget_bid_price_for_plot)
    # print(bitget_ask_price_for_plot)
    if i >= max_size:
        i = 0

def plot_cont(xmax, symbol):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fig.suptitle(symbol + "USDT")
    def update(i):
        x = range(len(bitget_ask_price_for_plot))
        ax.clear()
        mutex.acquire()
        ax.plot(x, bitget_ask_price_for_plot, color='blue')
        ax.plot(x, bitget_bid_price_for_plot, color='pink')
        ax.plot(x, bnb_ask_price_for_plot, linestyle='dashed', color='red')
        ax.plot(x, bnb_bid_price_for_plot, linestyle='dashed', color='green')
        # ax.scatter(x, arbitrage_opportunity, marker="*", color='gold')
        mutex.release()

    a = anim.FuncAnimation(fig, update, frames=xmax, repeat=True)
    plt.show(block=True)
    return a

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='symbol')
    parser.add_argument('-symbol', help="symbol", required=True)
    args = parser.parse_args()

    symbol = args.symbol

    bnb = BINANCE(symbol)
    bitget = BITGET(symbol)
    okex = OKEX(symbol)

    # threading.Thread(target=bnb.start_ws, args=("bookTicker", store_value)).start()
    # threading.Thread(target=bitget.start_ws, args=("books1", store_value)).start()
    threading.Thread(target=okex.start_ws, args=("books")).start()
    # threading.Thread(target=store_value, args=(symbol, )).start()
    # threading.Thread(target=plot_cont, args=(max_size, symbol+'USDT')).start()
    plot_cont(max_size, symbol+'USDT')