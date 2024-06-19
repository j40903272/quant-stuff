from enum import Enum

# transaction costs and tick sizes in Binance and Ftx


# transaction costs
class BinanceTx(Enum):
    DEFAULT = 0.0004
    BTCUSDT_UPERP = 0.0004
    ETHUSDT_UPERP = 0.0004
    BNBUSDT_UPERP = 0.0004


class FtxTx(Enum):
    DEFAULT = 0.0007
    BTCUSDT_UPERP = 0.0007
    ETHUSDT_UPERP = 0.0007
    BNBUSDT_UPERP = 0.0007


# tick sizes
class BinanceTickSize(Enum):
    DEFAULT = 0.01
    BTCUSDT_UPERP = 0.01
    ETHUSDT_UPERP = 0.01
    BNBUSDT_UPERP = 0.01


class FtxTickSize(Enum):
    DEFAULT = 0.01
    BTCUSDT_UPERP = 0.01
    ETHUSDT_UPERP = 0.01
    BNBUSDT_UPERP = 0.01


class ExchangeTransCostMethod(Enum):
    BINANCE = "ratio"
    FTX = "ratio"


class ExchangeEnum:
    def __init__(self, symbol, exchange) -> None:
        if len(sym_split_list := symbol.split('_')) > 2:
            symbol = '_'.join(sym_split_list[:-1])

        if exchange == "BINANCE":

            self.tx = BinanceTx[symbol].value if symbol in BinanceTx.__members__ else BinanceTx['DEFAULT'].value
            self.tick_size = BinanceTickSize[symbol].value if symbol in BinanceTickSize.__members__ else BinanceTickSize['DEFAULT'].value
            self.trans_cost_method = ExchangeTransCostMethod.BINANCE.value

        elif exchange == "FTX":
            self.tx = FtxTx[symbol].value if symbol in FtxTx.__members__ else FtxTx['DEFAULT'].value
            self.tick_size = FtxTickSize[symbol].value if symbol in FtxTickSize.__members__ else FtxTickSize['DEFAULT'].value
            self.trans_cost_method = ExchangeTransCostMethod.FTX.value

        # add your new exchange here if necessary


class StrategySettings:
    def __init__(self):
        pass

    @staticmethod
    def get_pf_param(symbol, exchange, position_scale=None, compound=False):
        enums = ExchangeEnum(symbol=symbol, exchange=exchange)
        param_dict = {
            "multiplier": 1,
            "capital": 1000,
            "transaction_cost": enums.tx,
            "trans_cost_method": enums.trans_cost_method,
            "overnight_cost_ratio": None,
            "tick_size": enums.tick_size,
        }
        if position_scale:
            param_dict['position_scale'] = position_scale
        if compound:
            param_dict['compound'] = compound

        return param_dict
