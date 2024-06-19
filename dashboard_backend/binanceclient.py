from binance.client import Client
from binance.exceptions import BinanceAPIException
import json, config
from flask import Flask, request, abort
import time
from time import strftime, localtime
import requests
import sqlite3

def epoch2dt(epoch_time):
    return strftime('%Y-%m-%d %H:%M:%S', localtime(epoch_time // 1000))

class BinanceClient:
    def __init__(self, name=None):
        api_key = config.API_KEY
        api_secret= config.API_SECRET
        self.client = Client(api_key=api_key, api_secret=api_secret)
        self.name = name

    def send_discord_message(self, message, name=None):
        payload = {
            'content': message
        }
        if self.name:
            payload['username'] = self.name
        if name:
            payload['username'] = name
        requests.post(config.DISCORD_WEBHOOK_URL, json=payload)

    def setup(self):
        self.client.futures_coin_change_position_mode(dualSidePosition='true') # hedge mode

    def _get_quantity_precision(self, symbol):
        info = self.client.futures_exchange_info()
        for x in info['symbols']:
            if x['symbol'] == symbol:
                return x['quantityPrecision']

    def _get_price_precision(self, symbol):
        info = self.client.futures_exchange_info()
        for x in info['symbols']:
            if x['symbol'] == symbol:
                return x['pricePrecision']

    def get_price(self, symbol):
        return float(self.client.futures_symbol_ticker(symbol=symbol)['price'])

    def margin_to_quantity(self, symbol, margin, leverage=1):
        self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
        return round(margin * leverage / self.get_price(symbol), self._get_quantity_precision(symbol))

    def get_open_positions(self):
        positions = self.client.futures_account()['positions']
        quantity = 0
        open_pos = []
        for pos in positions:
            if float(pos['positionAmt']) != 0:
                open_pos.append(pos)
        return open_pos

    def get_position(self, symbol, position_side=None):
        positions = self.client.futures_account()['positions']
        quantity = 0
        for pos in positions:
            if pos['symbol'] == symbol and (pos['positionSide'] == position_side or position_side is None):
                return pos
        return None

    def _round_price(self, symbol, price):
        return round(float(price), self._get_price_precision(symbol))

    def _round_quantity(self, symbol, quantity):
        return round(float(quantity), self._get_quantity_precision(symbol))

    def open_position_check(self, symbol, margin, position_side, leverage=1, strategy_name=None):
        current_position = self.get_position(symbol)
        if float(current_position['positionAmt']) != 0:
            self.close_position(symbol, current_position['positionSide'])
        return self.open_position(symbol, margin, position_side, leverage=leverage, strategy_name=strategy_name)

    def open_position(self, symbol, margin, position_side, leverage=1, strategy_name=None):
        side = 'BUY' if position_side == 'LONG' else 'SELL'
        quantity = self.margin_to_quantity(symbol, margin, leverage)
        ret = self.client.futures_create_order(symbol=symbol, side=side, positionSide=position_side, type='MARKET', quantity=quantity)
        # self.send_discord_message(f'open {position_side} {symbol}', name=strategy_name)
        return ret
    def open_position_quantity(self, symbol, quantity, position_side, leverage=1, strategy_name=None):
        side = 'BUY' if position_side == 'LONG' else 'SELL'
        ret = self.client.futures_create_order(symbol=symbol, side=side, positionSide=position_side, type='MARKET', quantity=quantity)
        # self.send_discord_message(f'open {position_side} {symbol}', name=strategy_name)
        return ret

    def set_sl_price(self, symbol, position_side, sl_price):
        sl_price = self._round_price(symbol, sl_price)
        side = 'SELL' if position_side == 'LONG' else 'BUY'
        quantity = self._round_quantity(symbol, self.get_position(symbol, position_side)['positionAmt'])
        return self.client.futures_create_order(symbol=symbol, side=side, positionSide=position_side, type='STOP_MARKET', stopPrice=sl_price, closePosition='true')

    def set_sl_percentage(self, symbol, position_side, sl_perc):
        sl_perc = sl_perc / 100
        entry_price = float(self.get_position(symbol, position_side)['entryPrice'])
        side_mult = 1 if position_side == 'LONG' else -1
        sl_price = entry_price * (1 - sl_perc)
        return self.set_sl_price(symbol, position_side, sl_price)

    def set_tp_price(self, symbol, position_side, tp_price):
        tp_price = self._round_price(symbol, tp_price)
        side = 'SELL' if position_side == 'LONG' else 'BUY'
        quantity = self._round_quantity(symbol, self.get_position(symbol, position_side)['positionAmt'])
        return self.client.futures_create_order(symbol=symbol, side=side, positionSide=position_side, type='TAKE_PROFIT_MARKET', stopPrice=tp_price, closePosition='true')

    def set_tp_percentage(self, symbol, position_side, tp_perc):
        tp_perc = tp_perc / 100
        entry_price = float(self.get_position(symbol, position_side)['entryPrice'])
        side_mult = 1 if position_side == 'LONG' else -1
        tp_price = entry_price * (1 + tp_perc)
        return self.set_tp_price(symbol, position_side, tp_price)

    def close_position(self, symbol, position_side, strategy_name="手動收割機", quantity=None):
        side = 'SELL' if position_side == 'LONG' else 'BUY'
        quantity = self._round_quantity(symbol, self.get_position(symbol, position_side)['positionAmt']) if quantity is None else quantity
        for order in self.client.futures_get_open_orders():
            if order['symbol'] == symbol and order['positionSide'] == position_side:
                self.client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
        ret = self.client.futures_create_order(symbol=symbol, side=side, positionSide=position_side, type='MARKET', quantity=abs(quantity))
        time.sleep(1)
        order_id = ret['orderId']
        entry = {
            'symbol': symbol,
            'time': epoch2dt(ret['updateTime']),
            # aggregate statistics
            'avg_price': [],
            'pnl': 0,
            'qty': 0
        }
        for trade in self.client.futures_account_trades(symbol=symbol, orderId=order_id):
            entry['avg_price'].append(float(trade['price']))
            entry['pnl'] += float(trade['realizedPnl'])
            entry['qty'] += float(trade['qty'])
        if entry['avg_price']:
            entry['avg_price'] = sum(entry['avg_price']) / len(entry['avg_price'])
        else:
            entry['avg_price'] = 0
        self.insert_closed_position_data(epoch2dt(ret['updateTime']), symbol, entry['pnl'], entry['qty'], entry['avg_price'], position_side, strategy_name)
        # self.send_discord_message(f'close {position_side} {symbol}', name=strategy_name)
        return ret

    def insert_closed_position_data(self, time, symbol, pnl, quantity, avg_price, side, strategy):
        conn = sqlite3.connect('trading.db')
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO history (Time, TradingPair, PNL, Quantity, AvgPrice, Side, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (time, symbol, pnl, quantity, avg_price, side, strategy))
        conn.commit()
        conn.close()
    def close_partial_position(self, symbol, position_side, margin, leverage, strategy_name=None):
        side = 'SELL' if position_side == 'LONG' else 'BUY'
        quantity = self.margin_to_quantity(symbol, margin, leverage)
        for order in self.client.futures_get_open_orders():
            if order['symbol'] == symbol and order['positionSide'] == position_side:
                self.client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
        ret = self.client.futures_create_order(symbol=symbol, side=side, positionSide=position_side, type='MARKET', quantity=abs(quantity))
        # self.send_discord_message(f'close {position_side} {symbol}', name=strategy_name)
        return ret

    def close_all_positions(self):
        positions = self.get_open_positions()
        for pos in positions:
            self.close_position(pos['symbol'], pos['positionSide'])

    def close_short(self):
        positions = self.get_open_positions()
        for pos in positions:
            if pos['positionSide'] == 'SHORT':
                self.close_position(pos['symbol'], pos['positionSide'])

    def close_long(self):
        positions = self.get_open_positions()
        for pos in positions:
            if pos['positionSide'] == 'LONG':
                self.close_position(pos['symbol'], pos['positionSide'])

    def get_balance(self):
        balances = self.client.futures_account_balance()
        for balance in balances:
            if balance['asset'] == 'USDT':
                return balance['balance']
        return 0

    def get_available_balance(self):
        balances = self.client.futures_account_balance()
        for balance in balances:
            if balance['asset'] == 'USDT':
                return balance['availableBalance']
        return 0

    def get_history(self, **kwargs):
        order_history = self.client.futures_get_all_orders(**kwargs)
        history = {
            x['orderId']: {
                'side': x['positionSide'],
                'symbol': x['symbol'],
                'time': epoch2dt(x['time']),
                # aggregate statistics
                'avg_price': [],
                'pnl': 0,
                'qty': 0
            }
            for x in order_history
            if (1 if x['positionSide'] == 'LONG' else -1) * (1 if x['side'] == 'BUY' else -1) == -1
        }

        for hist in self.client.futures_account_trades():
            order_id = hist['orderId']
            if order_id in history:
                history[order_id]['avg_price'].append(float(hist['price']))
                history[order_id]['pnl'] += float(hist['realizedPnl'])
                history[order_id]['qty'] += float(hist['qty'])
        for hist in history.values():
            if hist['avg_price']:
                hist['avg_price'] = sum(hist['avg_price']) / len(hist['avg_price'])
            else:
                hist['avg_price'] = 0
        return sorted(list(history.values()), key=lambda x: x['time'], reverse=True)

    def get_funding_rate_history(self):
        return self.client.futures_income_history(incomeType='FUNDING_FEE')
    def get_account_data(self):
        return self.client.get_account()['balances']

if __name__ == '__main__':
    from pprint import pprint
    client = BinanceClient()

    client.open_position('ETHUSDT', 25, 'LONG')
    # client.open_position('XRPUSDT', 25, 'LONG')
    # client.close_long()
    # client.open_position("SXPUSDT", 25, "LONG")
    # client.open_position("SXPUSDT", 25, "LONG")
    # print(client.get_open_positions())
    # print(client.get_account_data())
    # ret = client.close_position("SXPUSDT","LONG")
    # print(client.client.futures_account_trades(symbol=ret['symbol'], orderId=ret['orderId']))
    # pprint(client.client.futures_income_history(incomeType='FUNDING_FEE'))
