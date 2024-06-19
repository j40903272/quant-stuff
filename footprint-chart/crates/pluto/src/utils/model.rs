use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
pub type PlutoEventSender = mpsc::Sender<PlutoEvent>;
pub type PlutoEventReceiver = mpsc::Receiver<PlutoEvent>;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Symbol {
    pub exchange: String,
    pub symbol: String,
    pub currency: Option<String>,
}

impl<'de> serde::Serialize for Symbol {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.to_string().serialize(serializer)
    }
}

// if called sedre::Deserialize, it will use Symbol::from_string
impl<'de> serde::Deserialize<'de> for Symbol {
    fn deserialize<D>(deserializer: D) -> Result<Symbol, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Symbol::from_string(&s))
    }
}

impl Symbol {
    pub fn new(exchange: &str, symbol: &str, currency: Option<&str>) -> Self {
        Self {
            exchange: exchange.to_string(),
            symbol: symbol.to_string(),
            currency: currency.map(|x| x.to_string()),
        }
    }

    pub fn to_string(&self) -> String {
        match &self.currency {
            Some(currency) => format!("{}:{}-{}", self.exchange, self.symbol, currency),
            None => format!("{}:{}", self.exchange, self.symbol),
        }
    }

    pub fn from_string(s: &str) -> Self {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() == 2 {
            let parts2: Vec<&str> = parts[1].split('-').collect();
            if parts2.len() == 2 {
                Self {
                    exchange: parts[0].to_string(),
                    symbol: parts2[0].to_string(),
                    currency: Some(parts2[1].to_string()),
                }
            } else {
                Self { exchange: parts[0].to_string(), symbol: parts[1].to_string(), currency: None }
            }
        } else {
            panic!("Invalid symbol string: {}", s);
        }
    }
}

#[derive(Debug, Clone, serde_tuple::Serialize_tuple, serde_tuple::Deserialize_tuple)]
pub struct OrderbookItem {
    pub price: f64,
    pub qty: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Orderbook {
    pub symbol: Symbol,
    pub timestamp_ms: u64,
    pub bids: Vec<OrderbookItem>,
    pub asks: Vec<OrderbookItem>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TradeData {
    pub symbol: Symbol,
    pub timestamp_ms: u64,
    pub price: f64,
    pub qty: f64,
    // 如果是 true，代表是成交在 bid，也就是紅色
    // 如果是 false，代表是成交在 ask，也就是綠色
    pub is_buyer_maker: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct UserOrderData {
    pub symbol: Symbol,
    pub timestamp_ms: u64,
    pub price: f64,
    pub cum_qty: f64,
    pub qty: f64,
    pub side: crate::OrderSide,
    pub status: crate::OrderStatus,
    pub order_id: String,
    pub order_type: crate::OrderType,
    pub position_side: crate::PositionSide,
    pub time_in_force: Option<crate::TimeInForce>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LastestKlineData {
    pub symbol: Symbol,
    pub timestamp_ms: u64,
    pub interval: Option<String>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, serde_tuple::Serialize_tuple, serde_tuple::Deserialize_tuple)]
pub struct KlineData {
    pub timestamp_ms: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub symbol: Symbol,
    pub interval: String,
    pub datas: Vec<KlineData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExternalExchangeError {
    MakerOrderExecutedImmediately(UserOrderData),
    CancelOrderNotFound(UserOrderData),
}

impl From<ExternalExchangeError> for PlutoEvent {
    fn from(e: ExternalExchangeError) -> Self {
        PlutoEvent::ExternalExchangeError(e)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event")]
pub enum PlutoEvent {
    PartialOrderbookUpdate(Orderbook),
    FullOrderbookUpdate(Orderbook),
    TradeUpdate(TradeData),
    UserOrderUpdate(UserOrderData),
    ExternalExchangeError(ExternalExchangeError),
    LastestKlineUpdate(LastestKlineData),
    KlineUpdate(Kline),
}
