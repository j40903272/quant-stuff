from apiflask import APIFlask, Schema, abort
from apiflask.fields import Integer, String, Number, List,Nested, UUID
from apiflask.validators import Length, OneOf

EXCHANGE = ['binance', 'bingx']
ORDER_TYPE = ['LIMIT', 'MARKET']
POSITION_SIDE = ['BOTH', 'LONG', "SHORT"]
SIDE = ['SELL', 'BUY']

class Crendtial(Schema):
    credential = String(required=True)

class rExchange_rSymbol_INPUT(Schema):
    exchange = String(required=True, validate = OneOf(EXCHANGE))
    symbol = String(required=True)

class rExchange_INPUT(Schema):
    exchange = String(required=True, validate = OneOf(EXCHANGE))

class rExchange_oSymbol_INPUT(Schema):
    exchange = String(required=True, validate = OneOf(EXCHANGE))
    symbol = String(required=False, default ='BTC-USDT')

class Cancel_Order_INPUT(Schema):
    exchange = String(required=True, validate = OneOf(EXCHANGE))
    symbol = String(required=True)
    orderId = String(required=True)

class markPrice(Schema):
    symbol = String()
    markPrice = Number()

class markPrice_OUTPUT(Schema):
    message = String()
    data = List(Nested(markPrice))

class fundingRate(Schema):
    fundingRate = Number()

class fundingRate_OUTPUT(Schema):
    message = String()
    data = Nested(fundingRate)

class GetBestOffer_INPUT(Schema):
    asks = Number()
    bids = Number()
    exchange = String(required=True, validate = OneOf(EXCHANGE))
    symbol = String()

class PlaceOrder_INPUT(Schema):
    exchange = String(required=True, validate = OneOf(EXCHANGE))
    symbol = String(required=True,)
    type = String(required=True, validate = OneOf(ORDER_TYPE))
    side = String(required=True, validate = OneOf(SIDE))
    positionSide = String(required=True, validate = OneOf(POSITION_SIDE))
    price = Number()
    quantity = Number(required=True)
    timeforce = String(default='GTC')

class GetOrder(Schema):
    orderId = Integer()
    positionSide = String()
    price = Number()
    qty = Number()
    side = String()
    symbol = String()
    type = String()

class GetOrders_OUTPUT(Schema):
    message = String()
    data = List(Nested(GetOrder))

class Symbols_OUTPUT(Schema):
    message = String()
    data = List(String())

class HistoricalOrder(Schema):
    status = String()
    symbol = String()
    side = String()
    positionSide = String()
    avgPrice = Number()
    qty = Number()
    time = String()
    commission = Number()
    leverage = Integer()
    profit = Number()

class HistoricalOrder_OUTPUT(Schema):
    message = String()
    data = List(Nested(HistoricalOrder))

class Position(Schema):
    entryPrice = Number()
    leverage = Integer()
    positionAmt = Number()
    unRealizedProfit = Number()
    time = Integer()
    symbol = String()
    side = String()
    positionSide = String()

class Positions_OUTPUT(Schema):
    message = String()
    data = List(Nested(Position))

class Balance(Schema):
    available_balance = Number()
    balance = Number()
    
class Balance_OUTPUT(Schema):
    message = String()
    data = Nested(Balance)

