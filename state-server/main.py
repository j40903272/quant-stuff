from apiflask import APIFlask  # step one
from flask import request, jsonify
from flask_cors import CORS
from state_server import StateServer
from schema import *
import uuid

app = APIFlask(
    __name__,
    title="StateServer API",
    version="1.0.0",
)
CORS(app)

state_server = StateServer()

#exchange_list = ['binance']
#exchange_list = ['bingx']

# strategies = [
#     'f01dd2c557c54e230045411cb6ed952d',
#     'bd65f59fc3205a75bd4e325ed508ab98',
#     'f3d08ca10bee2f7093fb890e9788ccae',
#     'f806ec2e1822911600814211af0dd9f1'
# ]
@app.route('/test', methods=['GET'])
@app.input({'symbol': String(load_default=False)}, location='query')
def test(query_data):

    query_data['symbol'] = query_data['symbol'].replace("-", "")

    state_server.get_historical_orders(query_data['symbol'])

    return {
        'status': 200,
        #'data': state_server.get_position(query_data['exchange'], query_data['symbol'])
    }  

@app.route('/balance', methods=['GET'])
@app.input(Crendtial, location='headers')
@app.input(rExchange_INPUT, location='query')
@app.output(Balance_OUTPUT)
@app.doc(tags=['GET'])
def get_balance(headers_data, query_data): 

    data = (state_server.get_balance(query_data['exchange']))

    return {
        'message': data['message'],
        'data': data['data']
    }

@app.route('/symbols', methods=['GET'])
@app.input(Crendtial, location='headers')
@app.input(rExchange_INPUT, location='query')
@app.output(Symbols_OUTPUT)
@app.doc(tags=['GET'])
def get_symbols(headers_data, query_data): 

    data = (state_server.get_symbols(query_data['exchange']))

    return {
        'message': data['message'],
        'data': data['data']
    }

@app.route('/pending_orders', methods=['GET'])
@app.input(Crendtial, location='headers')
@app.input(rExchange_oSymbol_INPUT, location='query')
@app.output(GetOrders_OUTPUT)
@app.doc(tags=['GET'])
def get_pending_orders(headers_data, query_data): 

    if "symbol" in query_data:
        query_data['symbol'] = query_data['symbol'].replace("-", "")
    else:
        query_data['symbol'] = None

    data = state_server.get_pending_orders(query_data['exchange'], query_data['symbol'])

    return {
        'message':data['message'],
        'data':data['data']
    }

@app.route('/markPrice', methods=['GET'])
@app.input(Crendtial, location='headers')
@app.input(rExchange_oSymbol_INPUT, location='query')
@app.output(markPrice_OUTPUT)
@app.doc(tags=['GET'])
def get_markPrice(headers_data, query_data): 

    if "symbol" in query_data:
        query_data['symbol'] = query_data['symbol'].replace("-", "")
    else:
        query_data['symbol'] = None

    data = state_server.get_markPrice(query_data['exchange'], query_data['symbol'])

    return {
        'message':data['message'],
        'data':data['data']
    }

@app.route('/fundingRate', methods=['GET'])
@app.input(Crendtial, location='headers')
@app.input(rExchange_rSymbol_INPUT, location='query')
@app.output(fundingRate_OUTPUT)
@app.doc(tags=['GET'])
def get_fundingRate(headers_data, query_data): 

    if "symbol" in query_data:
        query_data['symbol'] = query_data['symbol'].replace("-", "")

    data = state_server.get_fundingRate(query_data['exchange'], query_data['symbol'])

    return {
        'message':data['message'],
        'data':data['data']
    }


@app.route('/cancel_order', methods=['POST'])
@app.input(Crendtial, location='headers')
@app.input(Cancel_Order_INPUT, location='query')
@app.output({'message': String()})
@app.doc(tags=['POST'])
def cancel_order(headers_data, query_data): 

    if "symbol" in query_data:
        query_data['symbol'] = query_data['symbol'].replace("-", "")
    else:
        query_data['symbol'] = None

    data = state_server.cancel_order(query_data['exchange'], query_data['symbol'], query_data['orderId'])

    return {
        'message':data['message'],
    }

@app.route('/historical_orders', methods=['GET'])
@app.input(Crendtial, location='headers')
@app.input({'exchange': String(), 'symbol': String(), 'startTime': Number(), 'endTime': Number()}, location='query')
@app.output(HistoricalOrder_OUTPUT)
@app.doc(tags=['GET'])
def get_historical_orders(headers_data, query_data):
    if "symbol" in query_data:
        query_data['symbol'] = query_data['symbol'].replace("-", "")
    else:
        query_data['symbol'] = None

    if "startTime" in query_data and "endTime" in query_data:
        data = state_server.get_historical_orders(query_data['exchange'], query_data['symbol'], query_data['startTime'], query_data['startTime'])
    else:
        data = state_server.get_historical_orders(query_data['exchange'], query_data['symbol'])

    return {
        'message': data['message'],
        'data': data['data']
    }  

@app.route('/open_positions', methods=['GET'])
@app.input(Crendtial, location='headers')
@app.input(rExchange_oSymbol_INPUT, location='query')
@app.output(Positions_OUTPUT)
@app.doc(tags=['GET'])
def get_open_positions(headers_data,query_data):
    if "symbol" in query_data:
        query_data['symbol'] = query_data['symbol'].replace("-", "")
    else:
        query_data['symbol'] = None

    data = state_server.get_open_positions(query_data['exchange'], query_data['symbol'])

    return {
        'message':data['message'],
        'data':data['data']
    }

@app.route('/get_best_offer', methods=['GET'])
@app.input(Crendtial, location='headers')
@app.input(rExchange_rSymbol_INPUT, location='query')
@app.doc(tags=['GET'])
def get_best_offer(query_data):
    return state_server.get_best_offer(query_data['exchange'], query_data['symbol'].replace("-", ""))

@app.route('/place_order', methods=['POST'])
@app.input(Crendtial, location='headers')
@app.input(PlaceOrder_INPUT, location='query')
@app.doc(tags=['POST'])
def place_order(headers_data, query_data):

    if "symbol" in query_data:
        query_data['symbol'] = query_data['symbol'].replace("-", "")
    else:
        query_data['symbol'] = None

    if 'price' not in query_data:
        query_data['price'] = None

    if 'timeforce' not in query_data:
        query_data['timeforce'] = 'GTC'

    data = state_server.place_order(query_data['exchange'], query_data['symbol'], query_data['type']
                                    , query_data['side'], query_data['positionSide'], query_data['quantity']
                                    , query_data['price'], query_data['timeforce'])
    return {
        'message':data['message'],
    }

@app.route('/update_symbols', methods=['GET'])
@app.input(Crendtial, location='headers')
@app.input(rExchange_rSymbol_INPUT, location='query')
@app.doc(tags=['FUNCTIONAL'])
def update_symbols(headers_data, query_data):
    return state_server.update_symbols(query_data['exchange'], query_data['symbol'])

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=8000, debug= True)


