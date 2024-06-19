use crate::*;
use anyhow::Result;
use hmac::{Hmac, Mac};
use reqwest::Method;
use serde_json::Value;
use sha2::Sha256;
use std::collections::BTreeMap;

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

#[derive(Clone)]
pub struct BingxExchange {
    apikey: Option<String>,
    secret: Option<String>,
}

impl BingxExchange {
    pub fn new(apikey: Option<String>, secret: Option<String>) -> Self {
        BingxExchange { apikey, secret }
    }

    pub fn pluto_symbol_to_string(symbol: &Symbol) -> String {
        return symbol.clone().to_string().replace("BINGX-PS:", "").to_uppercase();
    }

    fn get_endpoint(&self) -> String {
        "https://open-api.bingx.com".into()
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
            request = request.header("X-BX-APIKEY", apikey);
        }

        let response = request.send().await?;
        let text = response.text().await?;
        let result: T = serde_json::from_str(&text)?;
        Ok(result)
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
            request = request.header("X-BX-APIKEY", apikey);
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

    pub async fn get_balance(&self) -> Result<BingxBalanceResponse> {
        let value: Value = self.send_get_request_with_signature("/openApi/swap/v2/user/balance", None).await?;

        match value["data"].as_object() {
            Some(data) => {
                let data = data["balance"].as_object().unwrap();
                let asset = data["asset"].as_str().unwrap().to_string();
                let balance = data["balance"].as_str().unwrap().parse::<f64>().unwrap();
                let equity = data["equity"].as_str().unwrap().parse::<f64>().unwrap();

                Ok(BingxBalanceResponse { asset, balance, equity })
            }
            None => Err(anyhow::anyhow!("Failed to get balance")),
        }
    }

    pub async fn place_order(&self, order: OrderRequest) -> Result<BingxPlaceOrderResponse> {
        let params = jsonvalue_to_btreemap(serde_json::to_value(order).unwrap());
        let value: Value = self.send_request_with_signature(Method::POST, "/openApi/swap/v2/trade/order", Some(params)).await?;

        if value["code"].as_i64().unwrap() != 0 {
            return Err(anyhow::anyhow!(format!("Failed to place order: {}", value)));
        }

        match value["data"].as_object() {
            Some(data) => {
                let data = data["order"].clone();
                let response: BingxPlaceOrderResponse = serde_json::from_value(data.clone()).unwrap();
                Ok(response)
            }
            None => Err(anyhow::anyhow!(format!("Failed to place order: {}", value))),
        }
    }

    pub async fn cancel_order(&self, order_id: String) -> Result<Value> {
        todo!()
    }

    pub async fn get_order(&self, get_order_request: GetOrderRequest) -> Result<BingxGetOrderResponse> {
        let params = jsonvalue_to_btreemap(serde_json::to_value(get_order_request).unwrap());
        let value: Value = self.send_request_with_signature(Method::GET, "/openApi/swap/v2/trade/order", Some(params)).await?;

        match value["data"].as_object() {
            Some(data) => {
                let mut data = data["order"].clone();

                if data["status"].as_str().unwrap() == "CANCELLED" {
                    data["status"] = Value::String("CANCELED".into());
                }
                if data["status"].as_str().unwrap() == "PENDING" {
                    data["status"] = Value::String("NEW".into());
                }

                let data: BingxGetOrderResponse = serde_json::from_value(data).unwrap();
                Ok(data)
            }
            None => Err(anyhow::anyhow!(format!("Failed to get order: {}", value))),
        }
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct BingxBalanceResponse {
    asset: String,
    balance: f64,
    equity: f64,
}

// Object {"order": Object {"activationPrice": Number(0), "clientOrderID": String(""), "closePosition": String(""), "orderId": Number(1792926084711321600), "positionSide": String("LONG"),
//  "price": Number(0), "priceRate": Number(0), "quantity": Number(0.0001),
// "reduceOnly": Bool(false), "side": String("BUY"), "stopGuaranteed": String(""),
//  "stopLoss": String(""), "stopPrice": Number(0), "symbol": String("BTC-USDT"), "takeProfit": String(""),
// "timeInForce": String("GTC"), "type": String("MARKET"), "workingType": String("MARK_PRICE")}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BingxPlaceOrderResponse {
    pub activation_price: f64,
    #[serde(rename = "clientOrderID")]
    pub client_order_id: String,
    pub close_position: String,
    pub order_id: i64,
    pub position_side: PositionSide,
    pub price: f64,
    pub price_rate: f64,
    pub quantity: f64,
    pub reduce_only: bool,
    pub side: OrderSide,
    pub stop_guaranteed: String,
    pub stop_loss: String,
    pub stop_price: f64,
    pub symbol: String,
    pub take_profit: String,
    pub time_in_force: TimeInForce,
    #[serde(rename = "type")]
    pub order_type: OrderType,
    pub working_type: String,
}

// Object {"advanceAttr": Number(0), "avgPrice": String("70688.2"), "clientOrderId": String(""),
// "commission": String("-0.003534"), "cumQuote": String("7"), "executedQty": String("0.0001"),
// "leverage": String("5X"), "onlyOnePosition": Bool(false), "orderId": Number(1792928327137234944), "orderType": String(""),
// "origQty": String("0.0001"), "positionID": Number(0), "positionSide": String("LONG"), "postOnly": Bool(false),
// "price": String("70690.3"), "profit": String("0.0000"), "reduceOnly": Bool(false), "side": String("BUY"), "status": String("FILLED"), "stopGuaranteed": String("false"),
// "stopLoss": Object {"price": Number(0), "quantity": Number(0), "stopGuaranteed": String(""), "stopPrice": Number(0), "type": String(""), "workingType": String("")},
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BingxGetOrderResponse {
    pub advance_attr: f64,
    #[serde(with = "string_or_float")]
    pub avg_price: f64,
    pub client_order_id: String,

    // 負的是扣錢，正的是返佣
    #[serde(with = "string_or_float")]
    pub commission: f64,

    #[serde(with = "string_or_float")]
    pub cum_quote: f64,

    #[serde(with = "string_or_float")]
    pub executed_qty: f64,

    pub leverage: String,
    pub only_one_position: bool,
    pub order_id: i64,
    #[serde(rename = "orderType")]
    pub order_type_ignore: String,

    #[serde(with = "string_or_float")]
    pub orig_qty: f64,
    #[serde(rename = "positionID")]
    pub position_id: i64,
    pub position_side: PositionSide,
    pub post_only: bool,

    #[serde(with = "string_or_float")]
    pub price: f64,

    #[serde(with = "string_or_float")]
    pub profit: f64,

    pub reduce_only: bool,
    pub side: OrderSide,
    pub status: OrderStatus,
    pub stop_guaranteed: String,
    pub stop_loss: Value,
    // "stopLossEntrustPrice": Number(0), "stopPrice": String(""), "symbol": String("BTC-USDT"),
    // "takeProfit": Object {"price": Number(0), "quantity": Number(0), "stopGuaranteed": String(""), "stopPrice": Number(0), "type": String(""), "workingType": String("")},
    // "takeProfitEntrustPrice": Number(0), "time": Number(1716302399000), "trailingStopDistance": Number(0), "trailingStopRate": Number(0), "triggerOrderId": Number(0),
    // "type": String("MARKET"), "updateTime": Number(1716302399000), "workingType": String("MARK_PRICE")}
    pub stop_loss_entrust_price: f64,
    pub stop_price: String,
    pub symbol: String,
    pub take_profit: Value,
    pub take_profit_entrust_price: f64,
    pub time: i64,
    pub trailing_stop_distance: f64,
    pub trailing_stop_rate: f64,
    pub trigger_order_id: i64,

    #[serde(rename = "type")]
    pub order_type: OrderType,
    pub update_time: i64,
    pub working_type: String,
}
