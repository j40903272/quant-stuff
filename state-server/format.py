import time
class Formatter():
    def to_binanace_position(self, response):
        # Check if a valid response was received
        if response:
            # Use list comprehension to create the result list
            result = [
                {
                    'symbol': item['symbol'],
                    'leverage': item['leverage'],
                    'entryPrice': item['entryPrice'],
                    'positionAmt': abs(float(item['positionAmt'])),
                    'side': 'BUY' if item['positionSide'] == 'LONG' else 'SELL',
                    'unRealizedProfit': item['unRealizedProfit'],
                    'time': int(item['updateTime']),
                    'margin':abs(float(item['positionAmt'])) * float(item['entryPrice']) / float(item['leverage']),
                    'positionSide': item['positionSide']
                }
                for item in response
            ]
            return result
        else:
            return []
        
    def to_binanace_order(self, response):
    # Check if a valid response was received
        if response:
            # Use list comprehension to create the result list
            result = [
                {
                    'symbol': item['symbol'],
                    'type': item['type'],
                    'side': item['side'],
                    'positionSide': item['positionSide'],
                    'price': item['price'],
                    'qty': item['origQty'],
                    'orderId': item['orderId']
                }
                for item in response
            ]
            return result
        else:
            return []
        
    def to_binance_historical_orders(self, response):
    # Check if a valid response was received
        if response:
            # Use list comprehension to create the result list
            result = [
                {
                    'symbol': item['symbol'],
                    'side': item['side'],
                    'positionSide': item['positionSide'],
                    'avgPrice': item['price'],
                    'time':item['time'],
                    'qty':item['qty'],
                    'commission':item['commission'],
                    'PNL':item['realizedPnl']
                }
                for item in response
            ]
            return result
        else:
            return []
        
    def to_bingx_position(self, response):
        if response:
            result = [
                {
                    'symbol': item['symbol'].replace('-',''),
                    'leverage': item['leverage'],
                    'entryPrice': item['avgPrice'],
                    'positionAmt': item['positionAmt'],
                    'side': 'BUY' if item['positionSide'] == 'LONG' else 'SELL',
                    'positionSide': item['positionSide'],
                    'unRealizedProfit': item['unrealizedProfit'],
                    'time': int(time.time()*1000),
                    'margin': item['initialMargin']
                }
                for item in response                 
            ]
            return result
        else:
            return []

    def to_bingx_order(self, response):
        if response:
            result = [{
                    'symbol': item['symbol'].replace('-',''),
                    'type': item['type'],
                    'side': item['side'],
                    'positionSide': item['positionSide'],
                    'price': item['price'],
                    'qty': item['origQty'],
                    'orderId': item['orderId']
                }
                for item in response   
            ]
            return result
        else:
            return []
        
    def to_bingx_historical_orders(self, response):
    # Check if a valid response was received
        if response:
            # Use list comprehension to create the result list
            result = [
                {   'status':item['status'],
                    'symbol': item['symbol'],
                    'side': item['side'],
                    'positionSide': item['positionSide'],
                    'avgPrice': item['avgPrice'],
                    'time':item['time'],
                    'qty':item['executedQty'],
                    'commission':item['commission'],
                    'leverage':item['leverage'].replace('X',''),
                    'PNL':item['profit']
                }
                for item in response
            ]
            return result
        else:
            return []
        
    def to_bingx_symbol(self, symbol):
        if symbol:
            formatted_symbol = symbol.upper()
            formatted_symbol = formatted_symbol.replace('USDT','-USDT')
            return formatted_symbol
        return None
        
