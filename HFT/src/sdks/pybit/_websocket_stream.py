import websocket
import threading
import time
import json
import hmac
import logging
import re
import copy
from . import HTTP


logger = logging.getLogger(__name__)


SUBDOMAIN_TESTNET = "stream-testnet"
SUBDOMAIN_MAINNET = "stream"
DOMAIN_MAIN = "bybit"
DOMAIN_ALT = "bytick"

INVERSE_PERPETUAL = "Inverse Perp"
USDT_PERPETUAL = "USDT Perp"
SPOT = "Spot"


class _WebSocketManager:
    def __init__(self, callback_function, ws_name,
                 test, domain="", api_key=None, api_secret=None,
                 ping_interval=30, ping_timeout=10,
                 restart_on_error=True, trace_logging=False):

        self.test = test
        self.domain = domain

        # Set API keys.
        self.api_key = api_key
        self.api_secret = api_secret

        self.callback = callback_function
        self.ws_name = ws_name
        if api_key:
            self.ws_name += " (Auth)"

        # Setup the callback directory following the format:
        #   {
        #       "topic_name": function
        #   }
        self.callback_directory = {}

        # Set ping settings.
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        # Other optional data handling settings.
        self.handle_error = restart_on_error

        # Enable websocket-client's trace logging for extra debug information
        # on the websocket connection, including the raw sent & recv messages
        websocket.enableTrace(trace_logging)

        # Set initial state, initialize dictionary and connect.
        self._reset()

    def _on_open(self):
        """
        Log WS open.
        """
        logger.debug(f"WebSocket {self.ws_name} opened.")

    def _on_message(self, message):
        """
        Parse incoming messages.
        """
        self.callback(json.loads(message))

    def _connect(self, url):
        """
        Open websocket in a thread.
        """

        # Set endpoint.
        subdomain = SUBDOMAIN_TESTNET if self.test else SUBDOMAIN_MAINNET
        domain = DOMAIN_MAIN if not self.domain else self.domain
        url = url.format(SUBDOMAIN=subdomain, DOMAIN=domain)
        self.endpoint = url

        self.public_v1_websocket = True if url.endswith("v1") else False
        self.public_v2_websocket = True if url.endswith("v2") else False
        self.private_websocket = True if url.endswith("/spot/ws") else False

        self.ws = websocket.WebSocketApp(
            url=url,
            on_message=lambda ws, msg: self._on_message(msg),
            on_close=self._on_close(),
            on_open=self._on_open(),
            on_error=lambda ws, err: self._on_error(err)
        )

        # Setup the thread running WebSocketApp.
        self.wst = threading.Thread(target=lambda: self.ws.run_forever(
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout
        ))

        # Configure as daemon; start.
        self.wst.daemon = True
        self.wst.start()

        # Attempt to connect for X seconds.
        retries = 10
        while retries > 0 and (not self.ws.sock or not self.ws.sock.connected):
            retries -= 1
            time.sleep(1)

        # If connection was not successful, raise error.
        if retries <= 0:
            self.exit()
            raise websocket.WebSocketTimeoutException("Connection failed.")

        # If given an api_key, authenticate.
        if self.api_key and self.api_secret:
            print('auth')
            self._auth()

    def _auth(self):
        """
        Authorize websocket connection.
        """

        # Generate expires.
        expires = int((time.time() + 1) * 1000)

        # Generate signature.
        _val = f"GET/realtime{expires}"
        signature = str(hmac.new(
            bytes(self.api_secret, "utf-8"),
            bytes(_val, "utf-8"), digestmod="sha256"
        ).hexdigest())

        # Authenticate with API.
        self.ws.send(
            json.dumps({
                "op": "auth",
                "args": [self.api_key, expires, signature]
            })
        )

    def _on_error(self, error):
        """
        Exit on errors and raise exception, or attempt reconnect.
        """

        if not self.exited:
            logger.error(
                f"WebSocket {self.ws_name} encountered error: {error}.")
            self.exit()

        # Reconnect.
        if self.handle_error:
            self._reset()
            self._connect(self.endpoint)

    def _on_close(self):
        """
        Log WS close.
        """
        logger.debug(f"WebSocket {self.ws_name} closed.")

    def _reset(self):
        """
        Set state booleans and initialize dictionary.
        """
        self.exited = False
        self.auth = False
        self.data = {}

    def exit(self):
        """
        Closes the websocket connection.
        """

        self.ws.close()
        while self.ws.sock:
            continue
        self.exited = True


class _WebSocketManager_rev:
    def __init__(self, callback_function, ws_name,
                 test, suffix="", api_key=None, api_secret=None,
                 ping_interval=30, ping_timeout=10,
                 restart_on_error=True, trace_logging=False):
        self.endpoint = f'wss://stream.bybit.com/v5/{suffix}' if not test \
            else f'wss://stream-testnet.bybit.com/v5/{suffix}'
        self.private_websocket = True if 'private' in suffix else False

        # Set API keys.
        self.api_key = api_key
        self.api_secret = api_secret

        self.callback = callback_function
        self.ws_name = ws_name
        if api_key:
            self.ws_name += " (Auth)"

        self.callback_directory = {}

        # Set ping settings.
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        # Other optional data handling settings.
        self.handle_error = restart_on_error

        # Enable websocket-client's trace logging for extra debug information
        # on the websocket connection, including the raw sent & recv messages
        websocket.enableTrace(trace_logging)

        # Set initial state, initialize dictionary and connect.
        self._reset()

    def _on_open(self):
        """
        Log WS open.
        """
        logger.debug(f"WebSocket {self.ws_name} opened.")

    def _on_message(self, message):
        """
        Parse incoming messages.
        """
        self.callback(json.loads(message))

    def _connect(self, suffix='public/spot'):
        """
        Open websocket in a thread.
        """

        self.ws = websocket.WebSocketApp(
            url=self.endpoint,
            on_message=lambda ws, msg: self._on_message(msg),
            on_close=self._on_close(),
            on_open=self._on_open(),
            on_error=lambda ws, err: self._on_error(err)
        )

        # Setup the thread running WebSocketApp.
        self.wst = threading.Thread(target=lambda: self.ws.run_forever(
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout
        ))

        # Configure as daemon; start.
        self.wst.daemon = True
        self.wst.start()

        # Attempt to connect for X seconds.
        retries = 10
        while retries > 0 and (not self.ws.sock or not self.ws.sock.connected):
            retries -= 1
            time.sleep(1)

        # If connection was not successful, raise error.
        if retries <= 0:
            self.exit()
            raise websocket.WebSocketTimeoutException("Connection failed.")

        # If given an api_key, authenticate.
        if self.api_key and self.api_secret:
            self._auth()

    def _auth(self):
        """
        Authorize websocket connection.
        """
        # Generate expires.
        expires = int((time.time() + 10) * 1000)

        # Generate signature.
        _val = f"GET/realtime{expires}"
        signature = str(hmac.new(
            bytes(self.api_secret, "utf-8"),
            bytes(_val, "utf-8"), digestmod="sha256"
        ).hexdigest())

        # Authenticate with API.
        self.ws.send(
            json.dumps({
                "op": "auth",
                "args": [self.api_key, expires, signature]
            })
        )

    def _on_error(self, error):
        """
        Exit on errors and raise exception, or attempt reconnect.
        """

        if not self.exited:
            logger.error(
                f"WebSocket {self.ws_name} encountered error: {error}.")
            self.exit()

        # Reconnect.
        if self.handle_error:
            self._reset()
            self._connect()
            self.re_subscribe()

    def _on_close(self):
        """
        Log WS close.
        """
        logger.debug(f"WebSocket {self.ws_name} closed.")

    def _reset(self):
        """
        Set state booleans and initialize dictionary.
        """
        self.exited = False
        self.auth = False
        self.data = {}

    def exit(self):
        """
        Closes the websocket connection.
        """

        self.ws.close()
        while self.ws.sock:
            continue
        self.exited = True


class _V5WebSocketManager(_WebSocketManager_rev):
    def __init__(self, ws_name, **kwargs):
        super().__init__(self._handle_incoming_message, ws_name, **kwargs)

        self._subscriptions = {}

        self.no_interval_topics = ["publicTrade", "tickers"]
        self.private_topics = ["position", "execution", "order", "wallet"]

        self.symbol_wildcard = "*"
        self.symbol_separator = "|"
    
    def sync_subscriptions(self, subscriptions):
        self._subscriptions = subscriptions

    def subscribe(self, topic, callback, subscription_args):
        self._set_callback(topic, callback)
        self.ws.send(
            json.dumps({
                "op": "subscribe",
                "args": subscription_args
            })
        )
    
    def re_subscribe(self):
        def sub(arg):
            self.ws.send(
                json.dumps({
                    "op": "subscribe",
                    "args": arg
                })
            )
        for key in self._subscriptions.keys():
            if key not in self.private_topics:
                for symbol in self._subscriptions[key]:
                    sub([f'{key}.{symbol}'])
            else:
                sub(self._subscriptions[key])
        

    def _handle_incoming_message(self, message):
        def is_auth_message():
            if message.get("op") == "auth":
                return True
            else:
                return False

        def is_subscription_message():
            if message.get("op") == "subscribe":
                return True
            else:
                return False

        def initialise_local_data():
            # Create self.data
            try:
                self.data[topic]
            except KeyError:
                self.data[topic] = []

        def process_delta_orderbook():
            initialise_local_data()
            self.data[message['topic']] = message["data"]

        def process_delta_instrument_info():
            initialise_local_data()

            # Record the initial snapshot.
            if "snapshot" in message["type"]:
                self.data[topic] = message["data"]

            # Make updates according to delta response.
            elif "delta" in message["type"]:
                # Update.
                for update in message["data"]["update"]:
                    for key, value in update.items():
                        self.data[topic][key] = value

        # Check auth
        if is_auth_message():
            # If we get successful futures auth, notify user
            if message.get("success") is True:
                logger.info(f"Authorization for {self.ws_name} successful.")
                self.auth = True
            # If we get unsuccessful auth, notify user.
            elif message.get("success") is False:
                print(message)
                logger.info(f"Authorization for {self.ws_name} failed. Please "
                             f"check your API keys and restart.")

        # Check subscription
        elif is_subscription_message():
            # If we get successful futures subscription, notify user
            if message.get("success") is True:
                pass
                # logger.info(f"Subscription successful.")
            # Futures subscription fail
            elif message.get("success") is False:
                response = message["ret_msg"]
                logger.error("Couldn't subscribe to topic."
                             f"Error: {response}.")

        else:
            topic = message["topic"]
            if "orderbook" in topic:
                process_delta_orderbook()
                callback_data = copy.deepcopy(message)
                callback_data["data"] = self.data[topic]
            elif "instrument_info" in topic:
                process_delta_instrument_info()
                callback_data = copy.deepcopy(message)
                callback_data["type"] = "snapshot"
                callback_data["data"] = self.data[topic]
            else:
                callback_data = message
            callback_function = self._get_callback(topic)
            callback_function(callback_data)

    def custom_topic_stream(self, topic, callback):
        return self.subscribe(topic=topic, callback=callback)

    def _extract_topic(self, topic_string):
        """
        Regex to return the topic without the symbol.
        """
        if topic_string in (self.private_topics+self.no_interval_topics):
            return topic_string
        
        topic_without_symbol = re.match(r".*(\..*|)(?=\.)", topic_string)

        return topic_without_symbol[0].split('.')[0]

    @staticmethod
    def _extract_symbol(topic_string):
        """
        Regex to return the symbol without the topic.
        """
        symbol_without_topic = re.search(r"(?!.*\.)[A-Z*|]*$", topic_string)
        return symbol_without_topic[0]

    def _check_callback_directory(self, topics):
        for topic in topics:
            if topic in self.callback_directory:
                raise Exception(f"You have already subscribed to this topic: "
                                f"{topic}")

    def _set_callback(self, topic, callback_function):
        topic = self._extract_topic(topic)
        self.callback_directory[topic] = callback_function

    def _get_callback(self, topic):
        topic = self._extract_topic(topic)
        return self.callback_directory[topic]

    def _pop_callback(self, topic):
        topic = self._extract_topic(topic)
        self.callback_directory.pop(topic)


def _identify_ws_method(input_wss_url, wss_dictionary):
    """
    This method matches the input_wss_url with a particular WSS method. This
    helps ensure that, when subscribing to a custom topic, the topic
    subscription message is sent down the correct WSS connection.
    """
    path = re.compile("(wss://)?([^/\s]+)(.*)")
    input_wss_url_path = path.match(input_wss_url).group(3)
    for wss_url, function_call in wss_dictionary.items():
        wss_url_path = path.match(wss_url).group(3)
        if input_wss_url_path == wss_url_path:
            return function_call


def _find_index(source, target, key):
    """
    Find the index in source list of the targeted ID.
    """
    return next(i for i, j in enumerate(source) if j[key] == target[key])


def _make_public_kwargs(private_kwargs):
    public_kwargs = copy.deepcopy(private_kwargs)
    public_kwargs.pop("api_key")
    public_kwargs.pop("api_secret")
    return public_kwargs
