import argparse
import json
import os
from Exchange.binance_exchange import BINANCE
from Exchange.bingx_exchange import BINGX
import configparser

parser = argparse.ArgumentParser(description="config path")
parser.add_argument("-config", help="config", required=True)
args = parser.parse_args()

config_file_name = args.config
with open(os.path.join("./", config_file_name)) as f:
    config = json.load(f)

api_key_ini = configparser.ConfigParser()
api_key_ini.read(filenames=config["api_key_path"])

binance = BINANCE(
    api_key_ini=api_key_ini['binance'],
    maker_cost=config["exchanges"]["binance"]["maker_cost"],
    taker_cost=config["exchanges"]["binance"]["taker_cost"],
    symbol=config["symbol"],
)
bingx = BINGX(
    api_key_ini=api_key_ini['bingx'],
    maker_cost=config["exchanges"]["bingx"]["maker_cost"],
    taker_cost=config["exchanges"]["bingx"]["taker_cost"],
    symbol=config["symbol"],
)


for asset in json.loads(json.dumps(binance.get_balance())):
    if asset['asset'] == 'USDT':
        # {'accountAlias': 'SgXqfWmYfWSgoCSg', 'asset': 'USDT', 'balance': '2000.01305312', 'crossWalletBalance': '2000.01305312', 'crossUnPnl': '0.00000000', 'availableBalance': '1989.98864312', 'maxWithdrawAmount': '1989.98864312', 'marginAvailable': True, 'updateTime': 1696517632875}
        print(f"Binance USDT Balance: {asset['balance']}, availableBalance: {asset['availableBalance']}.")
        print()

bingx_balance = json.loads(bingx.get_balance())
bingx_balance = bingx_balance['data']['balance']
print(f"Bingx USDT Balance: {bingx_balance['balance']}, equity: {bingx_balance['equity']}, unrealizedProfit: {bingx_balance['unrealizedProfit']}.")
print()

print(bingx.get_all_position())
print()

for position in json.loads(json.dumps(binance.get_all_position())):
    if float(position['positionAmt']) != 0:
        print(position)

print(binance.cancel_all_order())

# print(binance.place_order(quantity=19.3, direction="SELL", leverage=1, order_type="MARKET"))
# print(binance.place_order(quantity=0.2, direction="SELL", leverage=1, order_type="MARKET"))
# print(bingx.place_order(19, 'BUY', 'SHORT', 'MARKET'))
# print(bingx.place_order(0.2, 'SELL', 'SHORT', 'MARKET'))
# print(bingx.place_order(0.2, 'SELL', 'LONG', 'MARKET'))
