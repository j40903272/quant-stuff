class Exchange:
    def __init__(
        self,
        api_key_ini,
        API_URL,
        WS_URL,
        name,
        symbol,
        maker_cost,
        taker_cost,
        logger=None,
    ):
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
        self.logger = logger
        self.API_URL = API_URL
        self.WS_URL = WS_URL
        self.APIKEY = api_key_ini['key']
        self.SECRETKEY = api_key_ini['secret']
        self.name = name
        self.symbol = symbol
        self.client = None  # Exchange client object (initially None)
        self.live_price = None  # Real-time price information (initially None)

        self.long_order_status = {}
        self.long_order_status["status"] = "NO_ORDER"

        self.short_order_status = {}
        self.short_order_status["status"] = "NO_ORDER"

        self.orderbook = {}

        self.ask_1 = 0.0
        self.bid_1 = 0.0

        self.maker_cost = maker_cost
        self.taker_cost = taker_cost

    def on_message(self, ws, message, callback):
        """
        WebSocket message handling method.

        Args:
            ws: WebSocket connection object.
            message: Received message.
            callback: Callback function to process the message.
        """
        pass

    def on_ticker_message(self, ws, message, callback=None):
        """
        Handle ticker-type WebSocket messages.

        Args:
            ws: WebSocket connection object.
            message: Received ticker message.
            callback: Optional callback function to process the message.
        """
        pass

    def on_user_message(self, ws, message, callback=None):
        """
        Handle user-related WebSocket messages.

        Args:
            ws: WebSocket connection object.
            message: Received user message.
            callback: Optional callback function to process the message.
        """
        pass

    def on_open(self, ws):
        """
        Handle operations when the WebSocket connection is established.

        Args:
            ws: WebSocket connection object.
        """
        pass

    def on_error(self, ws, error):
        """
        Handle WebSocket errors.

        Args:
            ws: WebSocket connection object.
            error: Error information.
        """
        print(f"{self.name} WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """
        Handle operations when the WebSocket connection is closed.

        Args:
            ws: WebSocket connection object.
            close_status_code: Close status code.
            close_msg: Close message.
        """
        print(f"### {self.name} WebSocket Closed ###")
