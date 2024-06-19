import time
import hmac
import json
import logging
import requests
import hashlib
import urllib

from datetime import datetime as dt

from .exceptions import FailedRequestError, InvalidRequestError
from . import VERSION

# Requests will use simplejson if available.
try:
    from simplejson.errors import JSONDecodeError
except ImportError:
    from json.decoder import JSONDecodeError


class _HTTPManager:
    def __init__(self, endpoint=None, api_key=None, api_secret=None,
                 logging_level=logging.INFO, log_requests=False,
                 request_timeout=10, recv_window=5000, force_retry=False,
                 retry_codes=None, ignore_codes=None, max_retries=3,
                 retry_delay=3, referral_id=None, record_request_time=False):
        """Initializes the HTTP class."""

        # Set the endpoint.
        if endpoint is None:
            self.endpoint = "https://api.bybit.com"
        else:
            self.endpoint = endpoint

        # Setup logger.

        self.logger = logging.getLogger(__name__)

        if len(logging.root.handlers) == 0:
            #no handler on root logger set -> we add handler just for this logger to not mess with custom logic from outside
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                                   datefmt="%Y-%m-%d %H:%M:%S"
                                                   )
                                 )
            handler.setLevel(logging_level)
            self.logger.addHandler(handler)

        self.logger.debug("Initializing HTTP session.")
        self.log_requests = log_requests

        # Set API keys.
        self.api_key = api_key
        self.api_secret = api_secret

        # Set timeout.
        self.timeout = request_timeout
        self.recv_window = recv_window
        self.force_retry = force_retry
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Set whitelist of non-fatal Bybit status codes to retry on.
        if retry_codes is None:
            self.retry_codes = {10002, 10006, 30034, 30035, 130035, 130150}
        else:
            self.retry_codes = retry_codes

        # Set whitelist of non-fatal Bybit status codes to ignore.
        if ignore_codes is None:
            self.ignore_codes = set()
        else:
            self.ignore_codes = ignore_codes

        # Initialize requests session.
        self.client = requests.Session()
        self.client.headers.update(
            {
                "User-Agent": "pybit-" + VERSION,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

        # Add referral ID to header.
        if referral_id:
            self.client.headers.update({"Referer": referral_id})

        # If true, records and returns the request's elapsed time in a tuple
        # with the response body.
        self.record_request_time = record_request_time

    def _auth(self, method, params, recv_window):
        """
        Generates authentication signature per Bybit API specifications.

        Notes
        -------------------
        Since the POST method requires a JSONified dict, we need to ensure
        the signature uses lowercase booleans instead of Python's
        capitalized booleans. This is done in the bug fix below.

        """

        api_key = self.api_key
        api_secret = self.api_secret

        if api_key is None or api_secret is None:
            raise PermissionError("Authenticated endpoints require keys.")

        # Append required parameters.
        params["api_key"] = api_key
        params["recv_window"] = recv_window
        params["timestamp"] = int(time.time() * 10 ** 3)

        # Sort dictionary alphabetically to create querystring.
        _val = "&".join(
            [str(k) + "=" + str(v) for k, v in sorted(params.items()) if
             (k != "sign") and (v is not None)]
        )

        # Bug fix. Replaces all capitalized booleans with lowercase.
        if method == "POST":
            _val = _val.replace("True", "true").replace("False", "false")

        # Return signature.
        return str(hmac.new(
            bytes(api_secret, "utf-8"),
            bytes(_val, "utf-8"), digestmod="sha256"
        ).hexdigest())

    @staticmethod
    def _verify_string(params, key):
        if key in params:
            if not isinstance(params[key], str):
                return False
            else:
                return True
        return True

    def _gen_signature(self, timestamp, payload):
        param_str = str(timestamp) + self.api_key + str(self.recv_window) + payload
        hash = hmac.new(bytes(self.api_secret, "utf-8"),
                        param_str.encode("utf-8"), hashlib.sha256)
        signature = hash.hexdigest()
        return signature

    def _submit_request(self, method=None, path=None, query=None, auth=False):
        """
        Submits the request to the API.

        Notes
        -------------------
        We use the params argument for the GET method, and data argument for
        the POST method. Dicts passed to the data argument must be
        JSONified prior to submitting request.

        """

        if query is None:
            query = {}


        # Store original recv_window.
        # recv_window = self.recv_window

        # Bug fix: change floating whole numbers to integers to prevent
        # auth signature errors.
        if query is not None:
            for i in query.keys():
                if isinstance(query[i], float) and query[i] == int(query[i]):
                    query[i] = int(query[i])

        # Send request and return headers with body. Retry if failed.
        retries_attempted = self.max_retries
        req_params = None

        while True:

            retries_attempted -= 1
            if retries_attempted < 0:
                raise FailedRequestError(
                    request=f"{method} {path}: {req_params}",
                    message="Bad Request. Retries exceeded maximum.",
                    status_code=400,
                    time=dt.utcnow().strftime("%H:%M:%S")
                )

            retries_remaining = f"{retries_attempted} retries remain."

            # Authenticate if we are using a private endpoint.
            if auth:
                if method == "GET":
                    payload = urllib.parse.urlencode(query)
                else:
                    payload = json.dumps(query)

                timestamp = str(int(time.time() * 10 ** 3))
                signature = self._gen_signature(timestamp, payload)
                headers = {
                    'X-BAPI-API-KEY': self.api_key,
                    'X-BAPI-SIGN': signature,
                    'X-BAPI-SIGN-TYPE': '2',
                    'X-BAPI-TIMESTAMP': timestamp,
                    'X-BAPI-RECV-WINDOW': str(self.recv_window),
                    'Content-Type': 'application/json'
                }
            else:
                headers = {}
           
            # Prepare request; use "params" for GET and "data" for POST.
            if method == "GET":
                r = self.client.prepare_request(
                    requests.Request(method, path, params=query,
                                     headers=headers)
                )
            else:
                r = self.client.prepare_request(
                    requests.Request(method, path,
                                        data=payload,
                                     headers=headers)
                )

            # Attempt the request.
            try:
                s = self.client.send(r, timeout=self.timeout)

            # If requests fires an error, retry.
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError
            ) as e:
                if self.force_retry:
                    self.logger.error(f"{e}. {retries_remaining}")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise e

            # Convert response to dictionary, or raise if requests error.
            try:
                s_json = s.json()

            # If we have trouble converting, handle the error and retry.
            except JSONDecodeError as e:
                if self.force_retry:
                    self.logger.error(f"{e}. {retries_remaining}")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise FailedRequestError(
                        request=f"{method} {path}: {req_params}",
                        message="Conflict. Could not decode JSON.",
                        status_code=409,
                        time=dt.utcnow().strftime("%H:%M:%S")
                    )
            if("v5" in path) or ("v3" in path):
                return s_json
            
            
            # If Bybit returns an error, raise.
            if s_json["retCode"]:

                # Generate error message.
                error_msg = (
                    f"{s_json['retCode']} (ErrCode: {s_json['retCode']})"
                )

                # Set default retry delay.
                err_delay = self.retry_delay

                # Retry non-fatal whitelisted error requests.
                if s_json["retCode"] in self.retry_codes:

                    # 10002, recv_window error; add 2.5 seconds and retry.
                    if s_json["retCode"] == 10002:
                        error_msg += ". Added 2.5 seconds to recv_window"
                        recv_window += 2500

                    # 10006, ratelimit error; wait until rate_limit_reset_ms
                    # and retry.
                    elif s_json["retCode"] == 10006:
                        self.logger.error(
                            f"{error_msg}. Ratelimited on current request. "
                            f"Sleeping, then trying again. Request: {path}"
                        )

                        # Calculate how long we need to wait.
                        limit_reset = s_json["rate_limit_reset_ms"] / 1000
                        reset_str = time.strftime(
                            "%X", time.localtime(limit_reset)
                        )
                        err_delay = int(limit_reset) - int(time.time())
                        error_msg = (
                            f"Ratelimit will reset at {reset_str}. "
                            f"Sleeping for {err_delay} seconds"
                        )

                    # Log the error.
                    self.logger.error(f"{error_msg}. {retries_remaining}")
                    time.sleep(err_delay)
                    continue

                elif s_json["retCode"] in self.ignore_codes:
                    pass

                else:
                    raise InvalidRequestError(
                        request=f"{method} {path}: {req_params}",
                        message=s_json["retMsg"],
                        status_code=s_json["retCode"],
                        time=dt.utcnow().strftime("%H:%M:%S")
                    )
            else:
                if self.record_request_time:
                    return s_json, s.elapsed
                else:
                    return s_json

    def api_key_info(self):
        """
        Get user's API key info.

        :returns: Request results as dictionary.
        """

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v2/private/account/api-key",
            auth=True
        )

class _UnifiedHTTPManager(_HTTPManager):
    """
        category: [spot, linearm inverse, option]
    """

    """
        ===================== Market =====================
    """
    def kline(self, **kwargs):
        """
        Get the klines.

        :param kwargs:
            endpoint: kline, mark-price-kline, index-price-kline, premium-index-price-kline
            category,
            symbol,
            interval:
                1,3,5,15,30,60,120,240,360,720,D,M,W
            
            start,
            end,
            limit
        :returns: Request results as dictionary.
        """

        suffix = f"/v5/market/{kwargs['endpoint']}"
        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs
        )
    
    def info(self, **kwargs):
        """
        Get the instruments info.

        :param kwargs:
            category,
            
            symbol,
            baseCoin,
            limit,
            cursor
        :returns: Request results as dictionary.
        """

        suffix = "/v5/market/instruments-info"
        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs
        )

    def get_orderbook(self, **kwargs):
        """
        Get the orderbook.

        :param kwargs:
            category,
            symbol,
            limit: 
                spot: [1, 50]
                linear&inverse: [1, 200]
                option: [1, 25]
        :returns: Request results as dictionary.
        """

        suffix = "/v5/market/orderbook"
        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs
        )

    def query_symbol(self, **kwargs):
        """
        Get the latest information for symbol.

        :param kwargs:
            category,
            symbol,
            baseCoin,
            expDate
        :returns: Request results as dictionary.
        """
        suffix = "/v5/market/tickers"
        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs
        )

    def funding_rates(self, **kwargs):
        """
        Gets the historical funding rate.
        if the interval is 8 hours and the current time is UTC 12, 
            then it returns the last funding rate, which settled at UTC 8. 
        
        To query the funding rate interval, please refer to instruments-info.

        :param kwargs:
            category,
            symbol,

            startTime,
            endTime,
            limit,
        :returns: Request results as dictionary.
        """

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v5/market/funding/history",
            query=kwargs
        )

    def open_interest(self, **kwargs):
        """
        Gets the total amount of unsettled contracts. In other words, the total
        number of contracts held in open positions.

        :param kwargs:
            category,
            symbol,
            intervalTime: 5min, 15min, 30min, 1h, 4h, 1d

            startTime,
            endTime,
            limit,
            cursor
        :returns: Request results as dictionary.
        """

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v5/market/open-interest",
            query=kwargs
        )

    def trades(self, **kwargs):
        """
        Query recent public trading data in Bybit.

        :param kwargs:
            category,
            symbol,

            baseCoin,
            optionType,
            limit,
        :returns: Request results as dictionary.
        """

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v5/market/recent-trade",
            query=kwargs
        )
    
    def insurances(self, **kwargs):
        """
        Query Bybit insurance pool data (BTC/USDT/USDC etc). The data is updated every 24 hours.

        :param kwargs:
            coin
        :returns: Request results as dictionary.
        """

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v5/market/insurance",
            query=kwargs
        )
    
    def risk_limit(self, **kwargs):
        """

        :param kwargs:
            category: linear

            symbol
        :returns: Request results as dictionary.
        """

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v5/market/risk-limit",
            query=kwargs
        )

    """
        ===================== Trade =====================
    """
    def execution(self, **kwargs):
        """
        :param kwargs:
            category,

            symbol,
            baseCoin,
            orderId,
            startTime,
            endTime,
            execType,
            limit:
                Limit for data size per page. [1, 100]. Default: 50
            cursor
        :returns: Request results as dictionary.
        """

        suffix = '/v5/execution/list'

        return self._submit_request(
            method='GET',
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def get_fee_rate(self, **kwargs):
        """
        Get the trading fee rate of derivatives.

        :param kwargs:
            category,
            
        :returns: Request results as dictionary.
        """

        suffix = "/v5/account/fee-rate"
        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )

    def place_order(self, **kwargs):
        """
        :param kwargs:
            category,
            symbol,
            side: 
                Buy, Sell
            orderType:
                Market, Limit
            qty

            isLeverage:
                Whether to borrow. Valid for spot only. 0(default): false, 1: true
            price,
            timeInForce:
                GTC, IOC, FOK, PostOnly
            reduceOnly:
                True, False
        :returns: Request results as dictionary.
        """

        suffix = '/v5/order/create'

        return self._submit_request(
            method='POST',
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def modify_order(self, **kwargs):
        """
        :covers: Linear contract / Option
        :param kwargs:
            category,
            symbol,
            
            orderId,
            qty,
            price
        :returns: Request results as dictionary.
        """

        suffix = '/v5/order/amend'

        return self._submit_request(
            method='POST',
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )

    def cancel_order(self, **kwargs):
        """
        :param kwargs:
            category,
            symbol,
            orderId
        :returns: Request results as dictionary.
        """

        suffix = '/v5/order/cancel'

        return self._submit_request(
            method='POST',
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def cancel_all_orders(self, **kwargs):
        """
        :param kwargs:
            category,
            
            symbol,
            baseCoin,
            settleCoin
        :returns: Request results as dictionary.
        """

        suffix = '/v5/order/cancel-all'

        return self._submit_request(
            method='POST',
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def get_order_hist(self, **kwargs):
        """
        :param kwargs:
            category,
            
            symbol,
            baseCoin,
            limit,
            cursor
        :returns: Request results as dictionary.
        """

        suffix = '/v5/order/history'

        return self._submit_request(
            method='GET',
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def get_open_order(self, **kwargs):
        """
        :param kwargs:
            category,

            symbol,
            baseCoin (option),
            settleCoin (linear),
            orderId,
            limit:
                Limit for data size per page. [1, 50]. Default: 20
            cursor
        :returns: Request results as dictionary.
        """

        suffix = '/v5/order/realtime'

        return self._submit_request(
            method='GET',
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def get_borrable(self, **kwargs):
        """
        :param kwargs:
            category(spot),
            symbol,
            side
        :returns: Request results as dictionary.
        """

        suffix = '/v5/order/spot-borrow-check'

        return self._submit_request(
            method='GET',
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def get_margin_loan_info(self, **kwargs):
        """
        Get Spot Interest & Quota
        
        :param kwargs:
            Parameter	Required	Type	Comment
            coin	true	string	currency
        :returns: Request results as dictionary.
        """
        suffix = "/spot/v3/private/cross-margin-loan-info"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )

    """
        ===================== Position =====================
    """
    def get_position(self, **kwargs):
        """
        :param kwargs:
            category (linear,option),
            
            baseCoin,
            settleCoin,
            limit:
                Limit for data size per page. [1, 200]. Default: 20
            cursor
        :returns: Request results as dictionary.
        """

        suffix = '/v5/position/list'

        return self._submit_request(
            method='GET',
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )

    def set_leverage(self, **kwargs):
        """
        :param kwargs:
            category (linear),
            symbol,
            buyLeverage,
            sellLeverage
        :returns: Request results as dictionary.
        """

        suffix = '/v5/position/set-leverage'

        return self._submit_request(
            method='POST',
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def set_risk_level(self, **kwargs):
        """
        :param kwargs:
            category (linear),
            symbol,
            riskId
        :returns: Request results as dictionary.
        """

        suffix = '/v5/position/set-risk-limit'

        return self._submit_request(
            method='POST',
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )

    """
        ===================== Account =====================
    """
    def get_wallet_balance(self, **kwargs):
        """
        Get wallet balance info.

        :param kwargs: 
            accountType: UNIFIED
        :returns: Request results as dictionary.
        """

        suffix = "/v5/account/wallet-balance"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def upgrade_uta(self, **kwargs):
        suffix = "/v5/account/upgrade-to-uta"

        return self._submit_request(
            method="POST",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )

    def get_borrow_history(self, **kwargs):
        """
        Get wallet balance info.

        :param kwargs: 
            currency,
            startTime,
            endTime,
            limit:
                Limit for data size per page. [1, 50]. Default: 20
            cursor
        :returns: Request results as dictionary.
        """

        suffix = "/v5/account/borrow-history"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )

    def get_collateral_info(self, **kwargs):
        """
        Get wallet balance info.

        :param kwargs: 
            currency
        :returns: Request results as dictionary.
        """

        suffix = "/v5/account/collateral-info"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        ) 
    
    def get_account(self, **kwargs):
        suffix = "/v5/account/info"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def get_transactions(self, **kwargs):
        """
        :param kwargs: 
            accountType: UNIFIED
            category
            type:
                TRADE, SETTLEMENT, FEE_REFUND, INTEREST
        :returns: Request results as dictionary.
        """
        suffix = "/v5/account/transaction-log"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def set_margin_mode(self, **kwargs):
        """
        Set margin mode

        :param kwargs: 
            setMarginMode:
                REGULAR_MARGIN, PORTFOLIO_MARGIN
        :returns: Request results as dictionary.
        """

        suffix = "/v5/account/set-margin-mode"

        return self._submit_request(
            method="POST",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )   

    """
        ===================== Asset =====================
    """
    def get_greeks(self, **kwargs):
        """
        Get wallet balance info.

        :param kwargs: 
            currency
        :returns: Request results as dictionary.
        """

        suffix = "/v5/asset/coin-greeks"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    

    def get_asset_info(self, **kwargs):
        """
        Get asset info.

        :param kwargs: 
            accountType(SPOT),
            coin
        :returns: Request results as dictionary.
        """

        suffix = "/v5/asset/transfer/query-asset-info"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def enable_transfer_coin_list(self, **kwargs):
        """
        enable_transfer_coin_list

        :param kwargs: 
            fromAccountType(UNIFIED)
            toAccountType(SPOT)
        :returns: Request results as dictionary.
        """

        suffix = "/v5/asset/transfer/query-transfer-coin-list"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def query_transfer_hist(self, **kwargs):
        """
        enable_transfer_coin_list

        :param kwargs: 
            transferId
        :returns: Request results as dictionary.
        """

        suffix = "/v5/asset/transfer/query-inter-transfer-list"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )
    
    def asset_transfer(self, **kwargs):
        """
        asset_transfer

        :param kwargs: 
            transferId
            coin
            amount
            fromAccountType(UNIFIED)
            toAccountType(SPOT)
        :returns: Request results as dictionary.
        """

        suffix = "/v5/asset/transfer/inter-transfer"

        return self._submit_request(
            method="POST",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )  

    def query_account_coin_balance(self, **kwargs):
        """
        query-account-coins-balance

        :param kwargs: 
            accountType
        :returns: Request results as dictionary.
        """

        suffix = "/v5/asset/transfer/query-account-coins-balance"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )

    """
        ===================== Margin trade =====================
    """
    def set_spot_margin_mode(self, **kwargs):
        """
        Set margin lvg

        :param kwargs: 
            spotMarginMode:
                1: 開啟，0: 關閉
        :returns: Request results as dictionary.
        """

        suffix = "/v5/spot-margin-trade/switch-mode"

        return self._submit_request(
            method="POST",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )   
    
    def set_margin_leverage(self, **kwargs):
        """
        Set margin lvg

        :param kwargs: 
            leverage:
                [2, 10]
        :returns: Request results as dictionary.
        """

        suffix = "/v5/spot-margin-trade/set-leverage"

        return self._submit_request(
            method="POST",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )   


class _FuturesHTTPManager(_HTTPManager):
    def orderbook(self, **kwargs):
        """
        Get the orderbook.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-orderbook.
        :returns: Request results as dictionary.
        """

        suffix = "/v2/public/orderBook/L2"
        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs
        )

    def latest_information_for_symbol(self, **kwargs):
        """
        Get the latest information for symbol.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-latestsymbolinfo.
        :returns: Request results as dictionary.
        """

        suffix = "/v2/public/tickers"
        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs
        )

    def query_symbol(self):
        """
        Get symbol info.

        :returns: Request results as dictionary.
        """

        suffix = "/v2/public/symbols"
        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix
        )

    def liquidated_orders(self, **kwargs):
        """
        Retrieve the liquidated orders. The query range is the last seven days
        of data.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-query_liqrecords.
        :returns: Request results as dictionary.
        """

        # Replace query param "from_id" since "from" keyword is reserved.
        # Temporary workaround until Bybit updates official request params
        if "from_id" in kwargs:
            kwargs["from"] = kwargs.pop("from_id")

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v2/public/liq-records",
            query=kwargs
        )

    def open_interest(self, **kwargs):
        """
        Gets the total amount of unsettled contracts. In other words, the total
        number of contracts held in open positions.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-marketopeninterest.
        :returns: Request results as dictionary.
        """

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v2/public/open-interest",
            query=kwargs
        )

    def latest_big_deal(self, **kwargs):
        """
        Obtain filled orders worth more than 500,000 USD within the last 24h.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-marketbigdeal.
        :returns: Request results as dictionary.
        """

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v2/public/big-deal",
            query=kwargs
        )

    def long_short_ratio(self, **kwargs):
        """
        Gets the Bybit long-short ratio.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-marketaccountratio.
        :returns: Request results as dictionary.
        """

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v2/public/account-ratio",
            query=kwargs
        )

    def query_trading_fee_rate(self, **kwargs):
        """
        Query trading fee rate.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-queryfeerate.
        :returns: Request results as dictionary.
        """
        return self._submit_request(
            method='GET',
            path=self.endpoint + '/v2/private/position/fee-rate',
            query=kwargs,
            auth=True
        )

    def lcp_info(self, **kwargs):
        """
        Get user's LCP (data refreshes once an hour). Only supports inverse
        perpetual at present. See
        https://bybit-exchange.github.io/docs/inverse/#t-liquidity to learn
        more.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-lcp.
        :returns: Request results as dictionary.
        """

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v2/private/account/lcp",
            query=kwargs,
            auth=True
        )

    def get_wallet_balance(self, **kwargs):
        """
        Get wallet balance info.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-balance.
        :returns: Request results as dictionary.
        """

        suffix = "/v2/private/wallet/balance"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs,
            auth=True
        )

    def wallet_fund_records(self, **kwargs):
        """
        Get wallet fund records. This endpoint also shows exchanges from the
        Asset Exchange, where the types for the exchange are
        ExchangeOrderWithdraw and ExchangeOrderDeposit.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-walletrecords.
        :returns: Request results as dictionary.
        """

        # Replace query param "from_id" since "from" keyword is reserved.
        # Temporary workaround until Bybit updates official request params
        if "from_id" in kwargs:
            kwargs["from"] = kwargs.pop("from_id")

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v2/private/wallet/fund/records",
            query=kwargs,
            auth=True
        )

    def withdraw_records(self, **kwargs):
        """
        Get withdrawal records.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-withdrawrecords.
        :returns: Request results as dictionary.
        """

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v2/private/wallet/withdraw/list",
            query=kwargs,
            auth=True
        )

    def asset_exchange_records(self, **kwargs):
        """
        Get asset exchange records.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-assetexchangerecords.
        :returns: Request results as dictionary.
        """

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v2/private/exchange-order/list",
            query=kwargs,
            auth=True
        )

    def server_time(self):
        """
        Get Bybit server time.

        :returns: Request results as dictionary.
        """

        suffix = "/v2/public/time"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix
        )

    def announcement(self):
        """
        Get Bybit OpenAPI announcements in the last 30 days by reverse order.

        :returns: Request results as dictionary.
        """

        return self._submit_request(
            method="GET",
            path=self.endpoint + "/v2/public/announcement"
        )



class _InverseFuturesHTTPManager(_FuturesHTTPManager):
    def query_kline(self, **kwargs):
        """
        Get kline.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-querykline.
        :returns: Request results as dictionary.
        """

        # Replace query param "from_time" since "from" keyword is reserved.
        # Temporary workaround until Bybit updates official request params
        if "from_time" in kwargs:
            kwargs["from"] = kwargs.pop("from_time")

        suffix = "/v2/public/kline/list"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs
        )

    def public_trading_records(self, **kwargs):
        """
        Get recent trades. You can find a complete history of trades on Bybit
        at https://public.bybit.com/.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-publictradingrecords.
        :returns: Request results as dictionary.
        """

        # Replace query param "from_id" since "from" keyword is reserved.
        # Temporary workaround until Bybit updates official request params
        if "from_id" in kwargs:
            kwargs["from"] = kwargs.pop("from_id")

        suffix = "/v2/public/trading-records"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs
        )

    def get_risk_limit(self, **kwargs):
        """
        Get risk limit.

        :param kwargs: See
            https://bybit-exchange.github.io/docs/inverse/#t-getrisklimit.
        :returns: Request results as dictionary.
        """

        suffix = "/v2/public/risk-limit/list"

        return self._submit_request(
            method="GET",
            path=self.endpoint + suffix,
            query=kwargs
        )
