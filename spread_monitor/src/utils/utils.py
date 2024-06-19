#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import datetime
from decimal import *


def find_closest(a, b, target):
    diff_a = abs(a - target)
    diff_b = abs(b - target)

    if diff_a < diff_b:
        return a
    else:
        return b

def get_iso_time():
    return datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()

def get_real_price(client, category, symbol, value, side):  # OK
    order_book = client.get_orderbook(symbol, category=category)[side]
    init_value = value
    total_amt = 0
    global_price = 0
    for order in order_book:
        price = float(order[0])
        amt = float(order[1])
        value_per_order = (price*amt)
        total_amt += amt
        value -= value_per_order
        global_price = price
        if(value < 0):
            break

    total_amt += (value/global_price)
    real_price = init_value/total_amt
    return real_price