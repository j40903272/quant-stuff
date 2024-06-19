from flask import Flask, jsonify, request
from binanceclient import BinanceClient
from flask_cors import CORS
import requests
import json
import sqlite3
app = Flask(__name__)
CORS(app)
client = BinanceClient()


@app.route('/open_positions', methods=['GET'])
def get_open_positions():
    try:
        positions = client.get_open_positions()
        return jsonify(positions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/balance', methods=['GET'])
def get_balance():
    try:
        balance = client.get_balance()
        return jsonify({"balance": balance}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/available_balance', methods=['GET'])
def get_available_balance():
    try:
        balance = client.get_available_balance()
        return jsonify({"available_balance": balance}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        conn = sqlite3.connect('trading.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history ORDER BY Time DESC")
        rows = cursor.fetchall()
        history = [{"time": row[0], "symbol": row[1], "pnl": row[2], "qty": row[3], "avg_price": row[4], "side": row[5]} for row in rows]
        conn.close()
        return jsonify(history), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/pnl_history', methods=['GET'])
def pnl_history():
    try:
        conn = sqlite3.connect('trading.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pnl_records")
        rows = cursor.fetchall()
        history = [{"time": row[0], "pnl": row[1]} for row in rows]
        conn.close()
        return jsonify(history), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/funding', methods=['GET'])
def get_funding():
    try:
        history = client.get_funding_rate_history()[::-1]
        return jsonify(history), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/close_all_positions', methods=['POST'])
def close_all_positions():
    try:
        client.close_all_positions()
        return jsonify({'message': 'All positions closed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/close_position', methods=['POST'])
def close_position():
    try:
        data = request.json
        symbol = data.get('symbol')
        position_side = data.get('position_side')
        print(client.close_position(symbol, position_side))
        if not symbol or not position_side:
            return jsonify({'error': 'Missing symbol or position side'}), 400

        result = client.close_position(symbol, position_side)
        return jsonify({'message': 'Position closed', 'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_funding', methods=['GET'])
def get_symbols():
    mark_price_data = requests.get('https://fapi.binance.com/fapi/v1/premiumIndex').json()
    return list(
        map(
            lambda x: (x['symbol'], float(x['lastFundingRate']) * 100),
            filter(
                lambda x: float(x['lastFundingRate']) * 100 <= -0.6,
                mark_price_data
            )
        )
    )

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.data
        string_data = data.decode('utf-8')
        string_data = string_data.rstrip(',\n}') + '}'
        data = json.loads(string_data)
        print(data)
        strategy_name = data['strategy'] if 'strategy' in data else None
        ticker = data['ticker'].split('.')[0]
        price = data['price']
        action = data['action'].replace(" ", "")
        action = action.replace(" ", "")
        margin = data['margin']
        leverage = data['leverage']
        if action == "openshort":
            new_position_side = 'SHORT'
            opposite_position_side = 'LONG'
            client.open_position_check(ticker, float(margin), new_position_side, leverage=int(leverage), strategy_name=strategy_name)
        elif action == "openlong":
            new_position_side = 'LONG'
            opposite_position_side = 'SHORT'
            client.open_position_check(ticker, float(margin), new_position_side, leverage=int(leverage), strategy_name=strategy_name)
        elif action == "closelong":
            client.close_position(ticker, 'LONG', strategy_name=strategy_name)
        elif action == "closeshort":
            client.close_position(ticker, 'SHORT', strategy_name=strategy_name)
        else:
            return jsonify({'error': 'Invalid action'}), 400

        # open_positions = client.get_open_positions()
        # for pos in open_positions:
        #     if pos['symbol'] == ticker and pos['positionSide'] == opposite_position_side:
        #         print(f"Closing opposite position: {opposite_position_side} for {ticker}")
        #         cp = client.close_position(ticker, opposite_position_side)
        #         print(cp)
        #         break
        # print(f"Opening new position: {new_position_side} for {ticker}")
        # open_position_response = client.open_position(ticker, float(margin), new_position_side, leverage = float(leverage))
        # print(open_position_response)



        return jsonify({'message': 'Webhook processed successfully'}), 200

    except json.JSONDecodeError as e:
        return jsonify({'error': 'Invalid JSON', 'details': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500

@app.route('/grid', methods=['POST'])
def grid():
    try:
        data = request.data
        string_data = data.decode('utf-8')
        string_data = string_data.rstrip(',\n}') + '}'
        data = json.loads(string_data)
        print(data)
        strategy_name = data['strategy'] if 'strategy' in data else None
        ticker = data['ticker'].split('.')[0]
        price = data['price']
        action = data['action'].replace(" ", "")
        action = action.replace(" ", "")
        margin = data['margin']
        quantity = float(data['quantity'])
        leverage = data['leverage']
        if action == "openshort":
            new_position_side = 'SHORT'
            opposite_position_side = 'LONG'
            client.open_position_quantity(ticker, quantity, new_position_side, leverage=int(leverage), strategy_name=strategy_name)
        elif action == "openlong":
            new_position_side = 'LONG'
            opposite_position_side = 'SHORT'
            client.open_position_quantity(ticker, quantity, new_position_side, leverage=int(leverage), strategy_name=strategy_name)
        elif action == "closelong":
            client.close_position(ticker, 'LONG', quantity=quantity, strategy_name=strategy_name)
        elif action == "closeshort":
            client.close_position(ticker, 'SHORT', quantity=quantity, strategy_name=strategy_name)
        else:
            return jsonify({'error': 'Invalid action'}), 400


        return jsonify({'message': 'Webhook processed successfully'}), 200

    except json.JSONDecodeError as e:
        return jsonify({'error': 'Invalid JSON', 'details': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500

@app.route('/testwebhook', methods=['POST'])
def testwebhook():
    try:
        data = request.data
        string_data = data.decode('utf-8')
        string_data = string_data.rstrip(',\n}') + '}'
        data = json.loads(string_data)
        # print(data)
        strategy_name = data['strategy'] if 'strategy' in data else None
        ticker = data['ticker'].split('.')[0]
        price = data['price']
        action = data['action'].replace(" ", "")
        action = action.replace(" ", "")
        margin = data['margin']
        leverage = data['leverage']
        print(data)

        # open_positions = client.get_open_positions()
        # for pos in open_positions:
        #     if pos['symbol'] == ticker and pos['positionSide'] == opposite_position_side:
        #         print(f"Closing opposite position: {opposite_position_side} for {ticker}")
        #         cp = client.close_position(ticker, opposite_position_side)
        #         print(cp)
        #         break
        # print(f"Opening new position: {new_position_side} for {ticker}")
        # open_position_response = client.open_position(ticker, float(margin), new_position_side, leverage = float(leverage))
        # print(open_position_response)



        return jsonify({'message': 'Webhook processed successfully'}), 200

    except json.JSONDecodeError as e:
        return jsonify({'error': 'Invalid JSON', 'details': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500


@app.route('/test', methods=['POST'])
def test():
    try:
        # client.close_position("ETHUSDT", "LONG")
        open_position_response = client.open_position("ETHUSDT", float("21"), "SHORT", leverage=1)
        return jsonify({'message': 'Webhook processed successfully'}), 200
    except json.JSONDecodeError as e:
        return jsonify({'error': 'Invalid JSON', 'details': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0')
