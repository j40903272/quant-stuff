from functools import partial
from binance.client import Client
import json
import websocket
from .exchange import Exchange
from binance.um_futures import UMFutures
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient


class BINANCE(Exchange):
    def __init__(
        self, api_key_ini, maker_cost, taker_cost, logger=None, symbol="ETH"
    ) -> None:
        """
        Initialize an exchange object.

        Args:
            logger: Logger object for logging.
            API_URL: API URL of the exchange.
            WS_URL: WebSocket URL of the exchange.
            APIKEY: API key for accessing the exchange's API.
            SECRETKEY: Secret key part of the API key for authentication.
            name: Name of the exchange.
            symbol: Currency pair traded on the exchange.
        """
        super().__init__(
            api_key_ini=api_key_ini,
            logger=logger,
            API_URL="https://api.binance.com",
            WS_URL="wss://fstream.binance.com/ws",    
            name="Binance",
            symbol=f"{symbol.lower()}usdt",
            maker_cost=maker_cost,
            taker_cost=taker_cost,
        )

        self.clients = Client(
            self.APIKEY,
            self.SECRETKEY,
        )

        self.get_asset_precision()

    def message_handler(self, _, message):
        print(message)

    def on_message(self, ws, message, callback):
        result = json.loads(message)
        result["exchange"] = "binance"
        callback(result)

    def on_ticker_message(self, ws, message, callback=None):
        msg = json.loads(message)
        if "c" in msg:  # check if the close price is in the message
            self.live_price = float(msg["c"])  # update the live price

        if callback != None:
            msg["exchange"] = "binance"
            callback(msg)

    def on_user_message(self, ws, message, callback=None):
        msg = json.loads(message)
        # print(msg)
        if msg["e"] == "ACCOUNT_UPDATE":
            return
        callback(msg)

    def start_ws(self, stream_name, callback=None):
        if stream_name == "!bookTicker":
            ws = websocket.WebSocketApp(
                f"wss://fstream.binance.com/ws{stream_name}",
                on_open=self.on_open,
                on_message=partial(self.on_message, callback=callback),
                on_error=self.on_error,
                on_close=self.on_close,
            )
            print("Binance ws subscribed to: ", f"{stream_name}")
        else:
            ws = websocket.WebSocketApp(
                f"wss://fstream.binance.com/ws/{self.symbol}@{stream_name}",
                on_open=self.on_open,
                on_message=partial(self.on_ticker_message, callback=callback),
                on_error=self.on_error,
                on_close=self.on_close,
            )
            print("Binance ws subscribed to: ", f"{self.symbol}@{stream_name}")
        ws.run_forever(reconnect=5)

    def start_user_ws(self, callback=None):
        listen_key = self.clients.futures_stream_get_listen_key()
        user_ws = websocket.WebSocketApp(
            f"wss://fstream.binance.com/ws/{listen_key}",
            on_open=self.on_open,
            on_message=partial(self.on_user_message, callback=callback),
            on_error=self.on_error,
            on_close=self.on_close,
        )
        user_ws.run_forever(reconnect=5)

    def get_all_position(self):
        return self.clients.futures_position_information()

    def get_asset_precision(self):
        # {'symbol': 'TRBUSDT', 'pair': 'TRBUSDT', 'contractType': 'PERPETUAL',
        # 'deliveryDate': 4133404800000, 'onboardDate': 1569398400000, 'status': 'TRADING',
        # 'maintMarginPercent': '2.5000', 'requiredMarginPercent': '5.0000', 'baseAsset': 'TRB',
        # 'quoteAsset': 'USDT', 'marginAsset': 'USDT', 'pricePrecision': 3, 'quantityPrecision': 1,
        # 'baseAssetPrecision': 8, 'quotePrecision': 8, 'underlyingType': 'COIN',
        # 'underlyingSubType': ['Oracle'], 'settlePlan': 0, 'triggerProtect': '0.1500',
        # 'liquidationFee': '0.015000', 'marketTakeBound': '0.15', 'maxMoveOrderLimit': 10000,
        # 'filters': [{'maxPrice': '5138.000', 'filterType': 'PRICE_FILTER', 'minPrice': '1.230',
        # 'tickSize': '0.001'}, {'maxQty': '1000000.0', 'minQty': '0.1', 'filterType': 'LOT_SIZE',
        # 'stepSize': '0.1'}, {'minQty': '0.1', 'maxQty': '2000.0', 'filterType': 'MARKET_LOT_SIZE',
        # 'stepSize': '0.1'}, {'filterType': 'MAX_NUM_ORDERS', 'limit': 200}, {'filterType':
        # 'MAX_NUM_ALGO_ORDERS', 'limit': 10}, {'filterType': 'MIN_NOTIONAL', 'notional': '5'},
        # {'multiplierDown': '0.8500', 'multiplierDecimal': '4', 'multiplierUp': '1.1500',
        # 'filterType': 'PERCENT_PRICE'}], 'orderTypes': ['LIMIT', 'MARKET', 'STOP', 'STOP_MARKET',
        # 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET', 'TRAILING_STOP_MARKET'], 'timeInForce': ['GTC', 'IOC', 'FOK', 'GTX', 'GTD']}
        exchange_info = self.clients.futures_exchange_info()
        for pair_info in exchange_info["symbols"]:
            if pair_info["symbol"] == self.symbol.upper():
                self.quantityPrecision = int(pair_info["quantityPrecision"])
                self.pricePrecision = int(pair_info["pricePrecision"])
                return
        raise ValueError(f"Precision not found for symbol {self.symbol}")

    def place_order(
        self,
        quantity,
        direction,
        leverage,
        price=None,
        stop_loss=None,
        take_profit=None,
        order_type="LIMIT",
        time_in_force="GTC",
    ):
        self.clients.futures_change_leverage(symbol=self.symbol, leverage=leverage)

        side = Client.SIDE_BUY if direction == "LONG" else Client.SIDE_SELL
        binance_order_type = (
            Client.ORDER_TYPE_LIMIT
            if order_type == "LIMIT"
            else Client.ORDER_TYPE_MARKET
        )

        # print(quantity, precision)
        # if self.live_price:
        #     # Calculate quantity based on the live price and position size in USDT
        #    quantity = round(position_size_usdt / self.live_price, precision)
        # else:
        #     print("Error: Live price not available.")
        #     return

        order_params = {
            "symbol": self.symbol,
            "side": side,
            "type": binance_order_type,
            "quantity": quantity,
        }

        if order_type == "LIMIT":
            order_params["price"] = round(price, self.pricePrecision)
            order_params["timeInForce"] = time_in_force
            print("pricePrecision", self.pricePrecision)
            # order_params["price"] = str(self.live_price)

        if stop_loss:
            order_params["stopPrice"] = stop_loss
        if take_profit:
            order_params["takeProfit"] = take_profit
        print(order_params)
        order = self.clients.futures_create_order(**order_params)
        return order

    def cancel_order(self, orderId):
        self.clients.futures_cancel_order(symbol=self.symbol, orderId=orderId)
        
    def get_balance(self):
        return self.clients.futures_account_balance()
    
    def get_all_position(self):
        return self.clients.futures_position_information()

    def cancel_all_order(self):
        return self.clients.futures_cancel_all_open_orders(symbol=self.symbol)

if __name__ == "__main__":
    # Create instance of BINANCE class
    binance_client = BINANCE()

    # Test parameters
    position_size_usdt = 10
    direction = "LONG"
    leverage = 10
    order_type = "LIMIT"

    response = binance_client.place_order(
        position_size_usdt, direction, leverage, order_type=order_type
    )

    print("Order Response:", response)
