use serde::{Deserialize, Serialize};
use serde_json::{from_value, Value};

pub(crate) mod string_or_float {
    use std::fmt;

    use serde::{de, Deserialize, Deserializer, Serializer};

    pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: fmt::Display,
        S: Serializer,
    {
        serializer.collect_str(value)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum StringOrFloat {
            String(String),
            Float(f64),
        }

        match StringOrFloat::deserialize(deserializer)? {
            StringOrFloat::String(s) => {
                if s == "INF" {
                    Ok(f64::INFINITY)
                } else {
                    s.parse().map_err(de::Error::custom)
                }
            }
            StringOrFloat::Float(i) => Ok(i),
        }
    }
}

pub(crate) mod string_or_float_opt {
    use std::fmt;

    use serde::{Deserializer, Serializer};

    pub fn serialize<T, S>(value: &Option<T>, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: fmt::Display,
        S: Serializer,
    {
        match value {
            Some(v) => crate::string_or_float::serialize(v, serializer),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Some(crate::string_or_float::deserialize(deserializer)?))
    }
}

pub(crate) mod string_or_bool {
    use std::fmt;

    use serde::{de, Deserialize, Deserializer, Serializer};

    pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: fmt::Display,
        S: Serializer,
    {
        serializer.collect_str(value)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<bool, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum StringOrFloat {
            String(String),
            Bool(bool),
        }

        match StringOrFloat::deserialize(deserializer)? {
            StringOrFloat::String(s) => s.parse().map_err(de::Error::custom),
            StringOrFloat::Bool(i) => Ok(i),
        }
    }
}

#[derive(PartialEq, Debug, Serialize, Deserialize, Clone)]
pub struct Bids {
    #[serde(with = "string_or_float")]
    pub price: f64,
    #[serde(with = "string_or_float")]
    pub qty: f64,
}

impl Bids {
    pub fn new(price: f64, qty: f64) -> Bids {
        Bids { price, qty }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Asks {
    #[serde(with = "string_or_float")]
    pub price: f64,
    #[serde(with = "string_or_float")]
    pub qty: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BinanceFuturesDepthOrderBookEvent {
    #[serde(rename = "e")]
    pub event_type: String,

    #[serde(rename = "E")]
    pub event_time: u64,

    #[serde(rename = "s")]
    pub symbol: String,

    #[serde(rename = "U")]
    pub first_update_id: u64,

    #[serde(rename = "u")]
    pub final_update_id: u64,

    #[serde(rename = "pu")]
    #[serde(default)]
    pub previous_final_update_id: Option<u64>,

    #[serde(rename = "b")]
    pub bids: Vec<Bids>,

    #[serde(rename = "a")]
    pub asks: Vec<Asks>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BinanceFuturesTradeEvent {
    #[serde(rename = "e")]
    pub event_type: String,

    #[serde(rename = "E")]
    pub event_time: u64,

    #[serde(rename = "T")]
    pub trade_time: u64,

    #[serde(rename = "s")]
    pub symbol: String,

    #[serde(rename = "t")]
    pub trade_id: u64,

    #[serde(rename = "p")]
    #[serde(with = "string_or_float")]
    pub price: f64,

    #[serde(rename = "q")]
    #[serde(with = "string_or_float")]
    pub qty: f64,

    #[serde(rename = "X")]
    pub execution_type: String,

    #[serde(rename = "m")]
    #[serde(with = "string_or_bool")]
    pub maker: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BinanceFuturesBookTickerEvent {
    #[serde(rename = "e")]
    pub event_type: String,

    #[serde(rename = "E")]
    pub event_time: u64,

    #[serde(rename = "s")]
    pub symbol: String,

    #[serde(rename = "T")]
    pub transaction_time: u64,

    #[serde(rename = "u")]
    pub final_update_id: u64,

    #[serde(rename = "b")]
    #[serde(with = "string_or_float")]
    pub best_bid_price: f64,

    #[serde(rename = "B")]
    #[serde(with = "string_or_float")]
    pub best_bid_qty: f64,

    #[serde(rename = "a")]
    #[serde(with = "string_or_float")]
    pub best_ask_price: f64,

    #[serde(rename = "A")]
    #[serde(with = "string_or_float")]
    pub best_ask_qty: f64,
}

// {
//     "e": "kline",     // Event type
//     "E": 1638747660000,   // Event time
//     "s": "BTCUSDT",    // Symbol
//     "k": {
//       "t": 1638747660000, // Kline start time
//       "T": 1638747719999, // Kline close time
//       "s": "BTCUSDT",  // Symbol
//       "i": "1m",      // Interval
//       "f": 100,       // First trade ID
//       "L": 200,       // Last trade ID
//       "o": "0.0010",  // Open price
//       "c": "0.0020",  // Close price
//       "h": "0.0025",  // High price
//       "l": "0.0015",  // Low price
//       "v": "1000",    // Base asset volume
//       "n": 100,       // Number of trades
//       "x": false,     // Is this kline closed?
//       "q": "1.0000",  // Quote asset volume
//       "V": "500",     // Taker buy base asset volume
//       "Q": "0.500",   // Taker buy quote asset volume
//       "B": "123456"   // Ignore
//     }
//   }
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BinanceFuturesKlineData {
    #[serde(rename = "t")]
    pub start_time: u64,

    #[serde(rename = "T")]
    pub close_time: u64,

    #[serde(rename = "s")]
    pub symbol: String,

    #[serde(rename = "i")]
    pub interval: String,

    #[serde(rename = "f")]
    pub first_trade_id: u64,

    #[serde(rename = "L")]
    pub last_trade_id: u64,

    #[serde(rename = "o")]
    #[serde(with = "string_or_float")]
    pub open: f64,

    #[serde(rename = "c")]
    #[serde(with = "string_or_float")]
    pub close: f64,

    #[serde(rename = "h")]
    #[serde(with = "string_or_float")]
    pub high: f64,

    #[serde(rename = "l")]
    #[serde(with = "string_or_float")]
    pub low: f64,

    #[serde(rename = "v")]
    #[serde(with = "string_or_float")]
    pub base_asset_volume: f64,

    #[serde(rename = "n")]
    pub number_of_trades: u64,

    #[serde(rename = "x")]
    pub is_kline_closed: bool,

    #[serde(rename = "q")]
    #[serde(with = "string_or_float")]
    pub quote_asset_volume: f64,

    #[serde(rename = "V")]
    #[serde(with = "string_or_float")]
    pub taker_buy_base_asset_volume: f64,

    #[serde(rename = "Q")]
    #[serde(with = "string_or_float")]
    pub taker_buy_quote_asset_volume: f64,

    #[serde(rename = "B")]
    pub ignore: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BinanceFuturesKlineEvent {
    #[serde(rename = "e")]
    pub event_type: String,

    #[serde(rename = "E")]
    pub event_time: u64,

    #[serde(rename = "s")]
    pub symbol: String,

    #[serde(rename = "k")]
    pub kline: BinanceFuturesKlineData,
}
