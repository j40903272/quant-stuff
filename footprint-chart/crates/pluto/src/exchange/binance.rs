use crate::*;
use anyhow::Result;
use hmac::{Hmac, Mac};
use reqwest::Method;
use sha2::Sha256;
use std::collections::BTreeMap;

#[derive(Clone)]
pub enum BinanceConfig {
    Spot,
    Futures,
    FuturesTestnet,
}

#[derive(Clone)]
pub struct BinanceExchange {
    config: BinanceConfig,
    apikey: Option<String>,
    secret: Option<String>,
}

fn jsonvalue_to_btreemap(json: serde_json::Value) -> BTreeMap<String, String> {
    let mut params: BTreeMap<String, String> = BTreeMap::new();
    for (key, value) in json.as_object().unwrap().iter() {
        match value {
            serde_json::Value::String(value) => {
                params.insert(key.to_string(), value.to_string());
            }
            serde_json::Value::Number(value) => {
                params.insert(key.to_string(), value.to_string());
            }
            _ => {}
        }
    }
    params
}

pub fn binance_ticker_to_future_symbol(ticker: String) -> Symbol {
    // 如果最後結束是 USDT, USDC, BTC 分別切
    if ticker.ends_with("USDT") {
        return Symbol::new("BINANCE-P", &ticker[0..ticker.len() - 4], Some("USDT"));
    } else if ticker.ends_with("USDC") {
        return Symbol::new("BINANCE-P", &ticker[0..ticker.len() - 4], Some("USDC"));
    } else if ticker.ends_with("BTC") {
        return Symbol::new("BINANCE-P", &ticker[0..ticker.len() - 3], Some("BTC"));
    } else {
        return Symbol::new("BINANCE-P", &ticker, None);
    }
}

impl BinanceExchange {
    pub fn new(config: BinanceConfig, apikey: Option<String>, secret: Option<String>) -> Self {
        Self { config, apikey, secret }
    }

    fn get_endpoint(&self) -> String {
        match self.config {
            BinanceConfig::Spot => "https://api.binance.com".to_string(),
            BinanceConfig::Futures => "https://fapi.binance.com".to_string(),
            BinanceConfig::FuturesTestnet => "https://testnet.binancefuture.com".to_string(),
        }
    }

    fn get_params_str(params: Option<BTreeMap<String, String>>, with_timestamp: bool) -> String {
        let mut params_str = String::new();

        if let Some(params) = params {
            params_str = params.iter().map(|(key, value)| format!("{}={}&", key, value)).collect::<String>();
        }

        if with_timestamp {
            let timestamp = chrono::Utc::now().timestamp_millis();
            params_str = format!("{}timestamp={}", params_str, timestamp);
        }

        if params_str.ends_with('&') {
            params_str.pop();
        }

        params_str
    }

    fn get_sign(params_str: String, secret: Option<String>) -> Option<String> {
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.unwrap().as_bytes()).unwrap();
        mac.update(params_str.as_bytes());
        let result = mac.finalize();
        let result = result.into_bytes();
        let result = hex::encode(result);
        Some(result)
    }

    pub async fn send_request_without_signature<T>(&self, method: Method, path: &str, params: Option<BTreeMap<String, String>>) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let endpoint = self.get_endpoint();
        let client = reqwest::Client::new();

        let params_str = Self::get_params_str(params.clone(), false);
        let url = format!("{}{}?{}", endpoint, path, params_str);
        let mut request = client.request(method, &url);

        if let Some(apikey) = &self.apikey {
            request = request.header("X-MBX-APIKEY", apikey);
        }

        let response = request.send().await?;
        let text = response.text().await?;
        let result: T = serde_json::from_str(&text)?;
        Ok(result)
    }

    pub async fn send_get_request_without_signature<T>(&self, path: &str, params: Option<BTreeMap<String, String>>) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        self.send_request_without_signature(Method::GET, path, params).await
    }

    pub async fn send_request_with_signature<T>(&self, method: Method, path: &str, params: Option<BTreeMap<String, String>>) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let endpoint = self.get_endpoint();
        let client = reqwest::Client::new();

        let mut params_str = String::new();
        params_str = Self::get_params_str(params.clone(), true);
        let signature = Self::get_sign(params_str.clone(), self.secret.clone());

        if let Some(signature) = signature {
            params_str = format!("{}&signature={}", params_str, signature);
        }

        let url = format!("{}{}?{}", endpoint, path, params_str);
        let mut request = client.request(method, &url);

        if let Some(apikey) = &self.apikey {
            request = request.header("X-MBX-APIKEY", apikey);
        }

        let response = request.send().await?;
        let text = response.text().await?;
        let result: T = serde_json::from_str(&text)?;
        Ok(result)
    }

    pub async fn send_get_request_with_signature<T>(&self, path: &str, params: Option<BTreeMap<String, String>>) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        self.send_request_with_signature(Method::GET, path, params).await
    }

    pub async fn send_post_request_with_signature<T>(&self, path: &str, params: Option<BTreeMap<String, String>>) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        self.send_request_with_signature(Method::POST, path, params).await
    }

    pub async fn send_delete_request_with_signature<T>(&self, path: &str, params: Option<BTreeMap<String, String>>) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        self.send_request_with_signature(Method::DELETE, path, params).await
    }

    pub async fn send_put_request_with_signature<T>(&self, path: &str, params: Option<BTreeMap<String, String>>) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        self.send_request_with_signature(Method::PUT, path, params).await
    }

    pub async fn get_balance(&self) -> Result<Vec<BinanceFuturesBalanceResponse>> {
        if let Ok(balance) = self.send_get_request_with_signature("/fapi/v2/balance", None).await {
            return Ok(balance);
        }
        Err(anyhow::anyhow!("Failed to get balance"))
    }

    pub async fn place_order(&self, order_request: OrderRequest) -> Result<Transaction> {
        let params = jsonvalue_to_btreemap(serde_json::to_value(&order_request)?);
        let value: serde_json::Value = self.send_post_request_with_signature("/fapi/v1/order", Some(params)).await?;
        let transaction: Transaction = match serde_json::from_value(value.clone()) {
            Ok(transaction) => transaction,
            Err(e) => {
                return Err(anyhow::anyhow!("parse responce json error {:?}, responce: {:?}", e, value));
            }
        };
        Ok(transaction)
    }

    pub async fn cancel_order(&self, order_cancel: OrderCancellation) -> Result<Transaction> {
        let params: BTreeMap<String, String> = jsonvalue_to_btreemap(serde_json::to_value(&order_cancel)?);
        let value: serde_json::Value = self.send_delete_request_with_signature("/fapi/v1/order", Some(params)).await?;
        let order_canceled: Transaction = match serde_json::from_value(value.clone()) {
            Ok(order_canceled) => order_canceled,
            Err(e) => {
                return Err(anyhow::anyhow!("parse responce json error {:?}, responce: {:?}", e, value));
            }
        };
        Ok(order_canceled)
    }

    pub async fn get_order(&self, get_order_request: GetOrderRequest) -> Result<Transaction> {
        let params = jsonvalue_to_btreemap(serde_json::to_value(&get_order_request)?);
        let value: serde_json::Value = self.send_get_request_with_signature("/fapi/v1/order", Some(params)).await?;
        let transaction: Transaction = match serde_json::from_value(value.clone()) {
            Ok(transaction) => transaction,
            Err(e) => {
                return Err(anyhow::anyhow!("parse responce json error {:?}, responce: {:?}", e, value));
            }
        };
        Ok(transaction)
    }

    pub async fn get_positions(&self) -> Result<Vec<FuturesPosition>> {
        let value: serde_json::Value = self.send_get_request_with_signature("/fapi/v2/positionRisk", None).await?;
        let positions: Vec<BinanceFuturesPositionResponse> = match serde_json::from_value(value.clone()) {
            Ok(positions) => positions,
            Err(e) => {
                return Err(anyhow::anyhow!("parse responce json error {:?}, responce: {:?}", e, value));
            }
        };
        let positions: Vec<FuturesPosition> = positions
            .into_iter()
            .map(|position| {
                let symbol = position.symbol.clone();
                let symbol = binance_ticker_to_future_symbol(symbol);
                let position = FuturesPosition {
                    symbol,
                    position_side: Some(position.position_side),
                    quantity: position.position_amt,
                    avg_price: Some(position.entry_price),
                    unrealized_profit: Some(position.unrealized_profit),
                    leverage: Some(position.leverage),
                    notional: Some(position.notional),
                    isolated: Some(position.isolated),
                };
                position
            })
            .collect();

        Ok(positions)
    }

    pub async fn get_userstream_listen_key(&self) -> Result<String> {
        let value: serde_json::Value = self.send_post_request_with_signature("/fapi/v1/listenKey", None).await?;
        let listen_key: String = match value["listenKey"].as_str() {
            Some(listen_key) => listen_key.to_string(),
            _ => {
                return Err(anyhow::anyhow!("parse responce json error, responce: {:?}", value));
            }
        };
        Ok(listen_key)
    }

    pub async fn keepalive_userstream_listen_key(&self, listen_key: &str) -> Result<String> {
        let mut params: BTreeMap<String, String> = BTreeMap::new();
        params.insert("listenKey".to_string(), listen_key.to_string());
        let value: serde_json::Value = self.send_put_request_with_signature("/fapi/v1/listenKey", Some(params)).await?;
        let listen_key: String = match value["listenKey"].as_str() {
            Some(listen_key) => listen_key.to_string(),
            _ => {
                return Err(anyhow::anyhow!("parse responce json error, responce: {:?}", value));
            }
        };
        Ok(listen_key)
    }
}

impl BinanceExchange {
    pub async fn get_klines(&self, symbol: &Symbol, interval: &str, limit: Option<u16>) -> Result<Kline> {
        let mut params: BTreeMap<String, String> = BTreeMap::new();
        params.insert("symbol".to_string(), Self::pluto_symbol_to_binance(symbol));
        params.insert("interval".to_string(), interval.to_string());
        if let Some(limit) = limit {
            params.insert("limit".to_string(), limit.to_string());
        }
        let value: serde_json::Value = self.send_get_request_without_signature("/fapi/v1/klines", Some(params)).await?;

        let klines: Vec<BinanceFuturesKlineResponse> = match serde_json::from_value(value.clone()) {
            Ok(klines) => klines,
            Err(e) => {
                return Err(anyhow::anyhow!("parse responce json error {:?}, responce: {:?}", e, value));
            }
        };
        let kline_datas = klines
            .into_iter()
            .map(|kline| KlineData {
                timestamp_ms: kline.open_time,
                open: kline.open,
                high: kline.high,
                low: kline.low,
                close: kline.close,
                volume: kline.volume,
            })
            .collect();

        let kline = Kline { symbol: symbol.clone(), interval: interval.to_string(), datas: kline_datas };
        return Ok(kline);
    }
}

// Some util functions
impl BinanceExchange {
    pub fn pluto_symbol_to_binance(symbol: &Symbol) -> String {
        return symbol.clone().to_string().replace("BINANCE-P:", "").replace("BINANCE:", "").replace("-", "").to_uppercase();
    }

    pub async fn get_order_by_id(&self, symbol: &Symbol, order_id: u64) -> Result<Transaction> {
        let get_order_request = GetOrderRequest {
            symbol: Self::pluto_symbol_to_binance(symbol),
            order_id: Some(order_id.to_string()),
            ..Default::default()
        };
        self.get_order(get_order_request).await
    }

    pub async fn get_order_by_client_id(&self, symbol: &Symbol, client_order_id: String) -> Result<Transaction> {
        let get_order_request = GetOrderRequest {
            symbol: Self::pluto_symbol_to_binance(symbol),
            orig_client_order_id: Some(client_order_id),
            ..Default::default()
        };
        self.get_order(get_order_request).await
    }

    pub async fn cancel_order_by_id(&self, symbol: &Symbol, order_id: String) -> Result<Transaction> {
        let order_cancel = OrderCancellation {
            symbol: Self::pluto_symbol_to_binance(symbol),
            order_id: Some(order_id.parse().unwrap()),
            ..Default::default()
        };
        self.cancel_order(order_cancel).await
    }

    // 給單向倉用，做多
    pub async fn buy_market(&self, symbol: &Symbol, quantity: f64) -> Result<Transaction> {
        let order_request = OrderRequest {
            symbol: Self::pluto_symbol_to_binance(symbol),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: Some(quantity),
            ..Default::default()
        };
        self.place_order(order_request).await
    }

    // 給單向倉用，做空
    pub async fn sell_market(&self, symbol: &Symbol, quantity: f64) -> Result<Transaction> {
        let order_request = OrderRequest {
            symbol: Self::pluto_symbol_to_binance(symbol),
            side: OrderSide::Sell,
            order_type: OrderType::Market,
            quantity: Some(quantity),
            ..Default::default()
        };
        self.place_order(order_request).await
    }

    pub async fn long_buy_limit(&self, symbol: &Symbol, price: f64, quantity: f64) -> Result<Transaction> {
        let order_request = OrderRequest {
            symbol: Self::pluto_symbol_to_binance(symbol),
            side: OrderSide::Buy,
            position_side: Some(PositionSide::Long),
            order_type: OrderType::Limit,
            quantity: Some(quantity),
            price: Some(price),
            time_in_force: Some(TimeInForce::GTC),
            ..Default::default()
        };
        self.place_order(order_request).await
    }

    pub async fn long_sell_limit(&self, symbol: &Symbol, price: f64, quantity: f64) -> Result<Transaction> {
        let order_request = OrderRequest {
            symbol: Self::pluto_symbol_to_binance(symbol),
            side: OrderSide::Sell,
            position_side: Some(PositionSide::Long),
            order_type: OrderType::Limit,
            quantity: Some(quantity),
            price: Some(price),
            time_in_force: Some(TimeInForce::GTC),
            ..Default::default()
        };
        self.place_order(order_request).await
    }

    pub async fn short_sell_limit(&self, symbol: &Symbol, price: f64, quantity: f64) -> Result<Transaction> {
        let order_request = OrderRequest {
            symbol: Self::pluto_symbol_to_binance(symbol),
            side: OrderSide::Sell,
            position_side: Some(PositionSide::Short),
            order_type: OrderType::Limit,
            quantity: Some(quantity),
            price: Some(price),
            time_in_force: Some(TimeInForce::GTC),
            ..Default::default()
        };
        self.place_order(order_request).await
    }

    pub async fn short_buy_limit(&self, symbol: &Symbol, price: f64, quantity: f64) -> Result<Transaction> {
        let order_request = OrderRequest {
            symbol: Self::pluto_symbol_to_binance(symbol),
            side: OrderSide::Buy,
            position_side: Some(PositionSide::Short),
            order_type: OrderType::Limit,
            quantity: Some(quantity),
            price: Some(price),
            time_in_force: Some(TimeInForce::GTC),
            ..Default::default()
        };
        self.place_order(order_request).await
    }

    pub async fn long_buy_limit_maker(&self, symbol: &Symbol, price: f64, quantity: f64) -> Result<Transaction> {
        let order_request = OrderRequest {
            symbol: Self::pluto_symbol_to_binance(symbol),
            side: OrderSide::Buy,
            position_side: Some(PositionSide::Long),
            order_type: OrderType::Limit,
            quantity: Some(quantity),
            price: Some(price),
            time_in_force: Some(TimeInForce::GTX),
            ..Default::default()
        };
        self.place_order(order_request).await
    }

    pub async fn long_sell_limit_maker(&self, symbol: &Symbol, price: f64, quantity: f64) -> Result<Transaction> {
        let order_request = OrderRequest {
            symbol: Self::pluto_symbol_to_binance(symbol),
            side: OrderSide::Sell,
            position_side: Some(PositionSide::Long),
            order_type: OrderType::Limit,
            quantity: Some(quantity),
            price: Some(price),
            time_in_force: Some(TimeInForce::GTX),
            ..Default::default()
        };
        self.place_order(order_request).await
    }

    pub async fn short_sell_limit_maker(&self, symbol: &Symbol, price: f64, quantity: f64) -> Result<Transaction> {
        let order_request = OrderRequest {
            symbol: Self::pluto_symbol_to_binance(symbol),
            side: OrderSide::Sell,
            position_side: Some(PositionSide::Short),
            order_type: OrderType::Limit,
            quantity: Some(quantity),
            price: Some(price),
            time_in_force: Some(TimeInForce::GTX),
            ..Default::default()
        };
        self.place_order(order_request).await
    }

    pub async fn short_buy_limit_maker(&self, symbol: &Symbol, price: f64, quantity: f64) -> Result<Transaction> {
        let order_request = OrderRequest {
            symbol: Self::pluto_symbol_to_binance(symbol),
            side: OrderSide::Buy,
            position_side: Some(PositionSide::Short),
            order_type: OrderType::Limit,
            quantity: Some(quantity),
            price: Some(price),
            time_in_force: Some(TimeInForce::GTX),
            ..Default::default()
        };
        self.place_order(order_request).await
    }

    pub async fn change_leverage(&self, symbol: &Symbol, leverage: u8) -> Result<BinanceFuturesChangeLeverageResponse> {
        let mut params: BTreeMap<String, String> = BTreeMap::new();
        params.insert("symbol".to_string(), Self::pluto_symbol_to_binance(symbol));
        params.insert("leverage".to_string(), leverage.to_string());
        let value: serde_json::Value = self.send_post_request_with_signature("/fapi/v1/leverage", Some(params)).await?;
        let response: BinanceFuturesChangeLeverageResponse = match serde_json::from_value(value.clone()) {
            Ok(response) => response,
            Err(e) => {
                return Err(anyhow::anyhow!("parse responce json error {:?}, responce: {:?}", e, value));
            }
        };
        Ok(response)
    }

    pub async fn change_position_mode(&self, dual_side_position: bool) -> Result<()> {
        let mut params: BTreeMap<String, String> = BTreeMap::new();
        params.insert("dualSidePosition".to_string(), dual_side_position.to_string());
        let value: serde_json::Value = self.send_post_request_with_signature("/fapi/v1/positionSide/dual", Some(params)).await?;
        match value["code"].as_i64() {
            Some(code) => {
                if code < 0 {
                    let msg = value["msg"].to_string();
                    if msg.contains("No need to change position side.") {
                        return Ok(());
                    } else {
                        return Err(anyhow::anyhow!(msg));
                    }
                } else {
                    return Ok(());
                }
            }
            None => {
                panic!("Unexpected response")
            }
        }
    }
}
