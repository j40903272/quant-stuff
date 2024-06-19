from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import configparser

from src.bot import Bot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path', required=True)
    args = parser.parse_args()
    config_file = args.config

    parser = configparser.ConfigParser()
    parser.read(config_file)

    config = {
        'symbol': parser.get("DEFAULT", "SYMBOL"),
        'leverage': parser.getint("DEFAULT", "LEVERAGE"),
        'precision': parser.getint("DEFAULT", "PRECISION"),
        'want_to_earn': parser.getfloat("DEFAULT", "WANT_TO_EARN"),
        'close_tolerant': parser.getfloat("DEFAULT", "CLOSE_TOLERANT"),
        'default_qty': parser.getfloat("DEFAULT", "DEFAULT_QTY"),
        'binance_maker_fee': parser.getfloat("DEFAULT", "BINANCE_MAKER_FEE"),
        'binance_taker_fee': parser.getfloat("DEFAULT", "BINANCE_TAKER_FEE"),
        'bingx_maker_fee': parser.getfloat("DEFAULT", "BINGX_MAKER_FEE"),
        'bingx_taker_fee': parser.getfloat("DEFAULT", "BINGX_TAKER_FEE"),
        'renew_interval_ms': parser.getint("DEFAULT", "RENEW_INTERVAL_MS"),
        'binance_apikey': os.getenv("BINANCE_APIKEY"),
        'binance_secret': os.getenv("BINANCE_SECRET"),
        'bingx_apikey': os.getenv("BINGX_APIKEY"),
        'bingx_secret': os.getenv("BINGX_SECRET"),
    }

    bot = Bot(config)
    bot.start()
