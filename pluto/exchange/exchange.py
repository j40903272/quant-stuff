import json
import time
import requests
import hmac
from hashlib import sha256

class Exchange():
    async def get_balance(self, asset):
        raise NotImplementedError("get_balance not implemented")

    async def get_available_balance(self, asset):
        raise NotImplementedError("get_available_balance not implemented")

    async def fetch_open_orders(self, symbol):
        raise NotImplementedError("fetch_open_orders not implemented")

    async def fetch_position(self, symbol):
        raise NotImplementedError("fetch_position not implemented")

    async def fetch_kline(self, symbol, interval):
        raise NotImplementedError("fetch_kline not implemented")

    async def fetch_ticker(self, symbol):
        raise NotImplementedError("fetch_ticker not implemented")

    async def fetch_lastest_price(self, symbol):
        raise NotImplementedError("fetch_lastest_price not implemented")

    async def fetch_trade_list(self, symbol):
        raise NotImplementedError("fetch_trade_list not implemented")

    async def place_limit_order(self, symbol, side, price, quantity):
        raise NotImplementedError("place_limit_order not implemented")

    async def place_order(self, symbol, type, positionSide, side, price, quantity):
        raise NotImplementedError("place_order not implemented")

    async def cancel_order(self, symbol, orderId):
        raise NotImplementedError("cancel_order not implemented")
    
    async def set_leverage(self, symbol, leverage):
        raise NotImplementedError("set_leverage not implemented")

    async def set_hedge_mode(self, on):
        raise NotImplementedError("set_hedge_mode not implemented")

class BinanceExchange(Exchange):
    def __init__(self, api_key, secret):
        self.api_key = api_key
        self.secret = secret

    def parse_param(self, paramsMap):
        sortedKeys = sorted(paramsMap)
        paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
        return paramsStr+"&timestamp="+str(int(time.time() * 1000))

    def get_sign(self, payload):
        signature = hmac.new(self.secret.encode("utf-8"), payload.encode("utf-8"), digestmod=sha256).hexdigest()
        return signature

    def send_request(self, method, path, urlpa, payload):
        url = "%s%s?%s&signature=%s" % ("https://fapi.binance.com", path, urlpa, self.get_sign(urlpa))
        headers = {
            "X-MBX-APIKEY" : self.api_key,
        }
        response = requests.request(method, url, headers=headers, data=payload)
        return response.text

    async def get_balance(self, asset = "USDT"):
        if asset != "USDT":
            raise Exception("BinanceExchange only support USDT")

        paramsStr = self.parse_param({})
        response = self.send_request("GET", "/fapi/v2/balance", paramsStr, {})
        json_data = json.loads(response)
        for item in json_data:
            if item["asset"] == "USDT":
                return float(item["balance"])
        raise Exception("BinanceExchange get_balance failed")

    async def get_available_balance(self, asset = "USDT"):
        if asset != "USDT":
            raise Exception("BinanceExchange only support USDT")

        paramsStr = self.parse_param({})
        response = self.send_request("GET", "/fapi/v2/balance", paramsStr, {})
        json_data = json.loads(response)
        for item in json_data:
            if item["asset"] == "USDT":
                return float(item["availableBalance"])
        raise Exception("BinanceExchange get_available_balance failed")

    async def fetch_open_orders(self, symbol):
        params = {}
        
        if symbol is not None:
            params["symbol"] = symbol

        response = self.send_request("GET", "/fapi/v1/openOrders", self.parse_param(params), {})
        json_data = json.loads(response)
        return json_data
    
    async def fetch_position(self, symbol=None):

        params = {}
        
        if symbol is not None:
            params["symbol"] = symbol

        response = self.send_request("GET", "/fapi/v2/positionRisk", self.parse_param(params), {})
        json_data = json.loads(response)
        return json_data

    async def fetch_trade_list(self, symbol):
        paramsStr = self.parse_param({"symbol": symbol, "limit": 1000})
        response = self.send_request("GET", "/fapi/v1/userTrades", paramsStr, {})
        json_data = json.loads(response)
        return json_data

    async def place_order(self, symbol, type, positionSide, side, price, quantity, timeinforce = "GTC"):
        data = {
            "symbol": symbol,
            "side": side,
            "positionSide": positionSide,
            "quantity": quantity,
            "type": type,
        }
        if type == "LIMIT":
            data["timeinforce"] = timeinforce
            data["price"] = price
        paramsStr = self.parse_param(data)
        response = self.send_request("POST", "/fapi/v1/order", paramsStr, {})
        json_data = json.loads(response)
        if json_data.get("code", 0) != 0:
            raise Exception("BinanceExchange place_order failed reason: " + json_data.get("msg", ""))
        return json_data
    
    async def history_orders(self, symbol, startTime = time.time() * 1000):
        data = {
            "symbol": symbol,
            "endTime":int(startTime),
            "startTime":int((startTime - (7 * 24 * 60 * 60 * 1000)))
        }
        paramsStr = self.parse_param(data)
        response = self.send_request("GET", "/fapi/v1/allOrders", paramsStr, {})
        json_data = json.loads(response)
        return json_data

    async def place_limit_tp_order(self, symbol, side, price, quantity):
        data = {
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": quantity,
            "type": "TAKE_PROFIT",
            "timeinforce": "GTC"
        }
        paramsStr = self.parse_param(data)
        response = self.send_request("POST", "/fapi/v1/order", paramsStr, {})
        json_data = json.loads(response)
        return json_data

    async def cancel_order(self, symbol, orderId):
        data = {
            "symbol": symbol,
            "orderId": orderId
        }
        paramsStr = self.parse_param(data)
        response = self.send_request("DELETE", "/fapi/v1/order", paramsStr, {})
        json_data = json.loads(response)
        return json_data
    
    async def set_leverage(self, symbol, leverage):
        data = {
            "symbol": symbol,
            "leverage": leverage
        }
        paramsStr = self.parse_param(data)
        response = self.send_request("POST", "/fapi/v1/leverage", paramsStr, {})
        json_data = json.loads(response)
        return json_data

    async def set_hedge_mode(self, on):
        data = {
            "dualSidePosition": "true" if on else "false"
        }
        paramsStr = self.parse_param(data)
        response = self.send_request("POST", "/fapi/v1/positionSide/dual", paramsStr, {})
        json_data = json.loads(response)
        return json_data

class BingxExchange(Exchange):
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key

    def parse_param(self, paramsMap):
        sortedKeys = sorted(paramsMap)
        paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
        return paramsStr+"&timestamp="+str(int(time.time() * 1000))

    def get_sign(self, payload):
        signature = hmac.new(self.secret_key.encode("utf-8"), payload.encode("utf-8"), digestmod=sha256).hexdigest()
        return signature

    def send_request(self, method, path, urlpa, payload):
        url = "%s%s?%s&signature=%s" % ("https://open-api.bingx.com", path, urlpa, self.get_sign(urlpa))
        headers = {
            'X-BX-APIKEY': self.api_key,
        }
        response = requests.request(method, url, headers=headers, data=payload)
        return response.text

    async def get_balance(self, asset = "USDT"):
        if asset != "USDT":
            raise Exception("BingxExchange only support USDT")
        paramsStr = self.parse_param({"recvWindow": 0})
        text = self.send_request("GET", "/openApi/swap/v2/user/balance", paramsStr, {})
        json_data = json.loads(text)
        return float(json_data["data"]["balance"]["balance"])
        # return float(json_data["data"]["balance"]["availableMargin"])

    async def get_available_balance(self, asset = "USDT"):
        if asset != "USDT":
            raise Exception("BingxExchange only support USDT")
        paramsStr = self.parse_param({"recvWindow": 0})
        text = self.send_request("GET", "/openApi/swap/v2/user/balance", paramsStr, {})
        json_data = json.loads(text)
        return float(json_data["data"]["balance"]["availableMargin"])

    async def fetch_open_orders(self, symbol):
        params = {}
        
        if symbol is not None:
            params["symbol"] = symbol

        text = self.send_request("GET", "/openApi/swap/v2/trade/openOrders", self.parse_param(params), {})
        json_data = json.loads(text)
        return json_data['data']['orders']
    
    async def fetch_position(self, symbol=None):

        params = {}
        
        if symbol is not None:
            params["symbol"] = symbol

        text = self.send_request("GET", "/openApi/swap/v2/user/positions", self.parse_param(params), {})
        json_data = json.loads(text)
        return json_data['data']

    async def fetch_kline(self, symbol, interval):
        paramsStr = self.parse_param({"symbol": symbol, "interval": interval, "limit": 1000})
        text = self.send_request("GET", "/openApi/swap/v3/quote/klines", paramsStr, {})
        json_data = json.loads(text)
        return json_data['data']

    async def fetch_ticker(self, symbol):
        paramsStr = self.parse_param({"symbol": symbol})
        text = self.send_request("GET", "/openApi/swap/v2/quote/ticker", paramsStr, {})
        json_data = json.loads(text)
        return json_data['data']

    async def fetch_lastest_price(self, symbol):
        paramsStr = self.parse_param({"symbol": symbol})
        text = self.send_request("GET", "/openApi/swap/v2/quote/price", paramsStr, {})
        json_data = json.loads(text)
        return json_data['data']['price']

    async def fetch_trade_list(self, symbol):
        paramsStr = self.parse_param({"symbol": symbol, "limit": 1000})
        text = self.send_request("GET", "/openApi/swap/v2/trade/allOrders", paramsStr, {})
        json_data = json.loads(text)
        orders = json_data['data']['orders']
        return [order for order in orders if order['status'] == 'FILLED' and order['symbol'] == symbol]

    async def place_limit_order(self, symbol, side, price, quantity, postOnly = True):
        data = {
            "symbol": symbol,
            "side": side,
            "price": price,
            "positionSide": "LONG" if side == "BUY" else "SHORT",
            "quantity": quantity,
            "type": "LIMIT",
            "timeInForce": "PostOnly" if postOnly else "GTC",
            "recvWindow": 0
        }
        paramsStr = self.parse_param(data)
        text = self.send_request("POST", "/openApi/swap/v2/trade/order", paramsStr, {})
        json_data = json.loads(text)
        return json_data

    async def place_order(self, symbol, type, positionSide, side, price, quantity):
        data = {
            "symbol": symbol,
            "side": side,
            "price": price,
            "positionSide": positionSide,
            "quantity": quantity,
            "type": type,
            "recvWindow": 0
        }
        paramsStr = self.parse_param(data)
        text = self.send_request("POST", "/openApi/swap/v2/trade/order", paramsStr, {})
        json_data = json.loads(text)
        if json_data.get("code", 0) != 0:
            raise Exception("BingxExchange place_order failed reason: " + json_data.get("msg", ""))
        return json_data['data']['order']
    
    async def history_orders(self, symbol, startTime = time.time() * 1000, limit = 500):
        data = {
            "symbol": symbol,
            "limit": 500,
            "startTime":int(startTime)
        }
        paramsStr = self.parse_param(data)
        response = self.send_request("GET", "/openApi/swap/v2/trade/allOrders", paramsStr, {})
        json_data = json.loads(response)
        print(json_data)
        orders = json_data['data']['orders']
        return [order for order in orders if order['status'] == 'FILLED' and order['symbol'] == symbol]

    async def close_position(self, symbol, side, price, quantity):
        data = {
            "symbol": symbol,
            "side": side,
            "price": price,
            "positionSide": "LONG" if side == "SELL" else "SHORT",
            "quantity": quantity,
            "type": "MARKET",
            "recvWindow": 0
        }
        paramsStr = self.parse_param(data)
        text = self.send_request("POST", "/openApi/swap/v2/trade/order", paramsStr, {})
        json_data = json.loads(text)
        return json_data

    async def place_limit_tp_order(self, symbol, side, price, quantity):
        data = {
            "symbol": symbol,
            "side": side,
            "price": price,
            "positionSide": "LONG" if side == "SELL" else "SHORT",
            "quantity": quantity,
            "type": "LIMIT",
            "timeInForce": "PostOnly",
            "recvWindow": 0
        }
        paramsStr = self.parse_param(data)
        text = self.send_request("POST", "/openApi/swap/v2/trade/order", paramsStr, {})
        json_data = json.loads(text)
        return json_data

    async def cancel_order(self, symbol, orderId):
        data = {
            "symbol": symbol,
            "orderId": orderId,
            "recvWindow": 0
        }
        paramsStr = self.parse_param(data)
        text = self.send_request("DELETE", "/openApi/swap/v2/trade/order", paramsStr, {})
        json_data = json.loads(text)
        return json_data

    async def set_leverage(self, symbol, leverage):
        data = {
            "symbol": symbol,
            "leverage": leverage,
            "side": "LONG",
            "recvWindow": 0
        }
        paramsStr = self.parse_param(data)
        text = self.send_request("POST", "/openApi/swap/v2/trade/leverage", paramsStr, {})
        json_data = json.loads(text)
        data["side"] = "SHORT"
        paramsStr = self.parse_param(data)
        text = self.send_request("POST", "/openApi/swap/v2/trade/leverage", paramsStr, {})
        json_data = json.loads(text)
        return json_data
