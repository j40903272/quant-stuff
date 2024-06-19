use crate::*;
use serde::{Deserialize, Serialize};

// [{
//     "accountAlias": "SgsRsRsRoCFzmY",
//     "asset": "USDC",
//     "availableBalance": "1000.20004000",
//     "balance": "1000.20004000",
//     "crossUnPnl": "0.00000000",
//     "crossWalletBalance": "1000.20004000",
//     "marginAvailable": true,
//     "maxWithdrawAmount": "1000.20004000",
//     "updateTime": 1713630374345
//   }
// ]
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BinanceFuturesBalanceResponse {
    pub account_alias: String,
    pub asset: String,
    #[serde(with = "string_or_float")]
    pub available_balance: f64,
    #[serde(with = "string_or_float")]
    pub balance: f64,
    #[serde(with = "string_or_float")]
    pub cross_un_pnl: f64,
    #[serde(with = "string_or_float")]
    pub cross_wallet_balance: f64,
    pub margin_available: bool,
    #[serde(with = "string_or_float")]
    pub max_withdraw_amount: f64,
    #[serde(rename = "updateTime")]
    pub update_time: u64,
}

// let order: OrderRequest = OrderRequest {
//     symbol: symbol.into(),
//     side: OrderSide::Sell,
//     position_side: None,
//     order_type: OrderType::Market,
//     time_in_force: None,
//     quantity: Some(qty.into()),
//     reduce_only: None,
//     price: None,
//     stop_price: None,
//     close_position: None,
//     activation_price: None,
//     callback_rate: None,
//     working_type: None,
//     price_protect: None,
//     new_client_order_id: None,
// };

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OrderSide {
    Buy,
    Sell,
}

/// By default, buy
impl Default for OrderSide {
    fn default() -> Self {
        Self::Buy
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OrderType {
    Limit,
    Market,
    Stop,
    StopMarket,
    TakeProfit,
    TakeProfitMarket,
    TrailingStopMarket,
}

/// By default, use market orders
impl Default for OrderType {
    fn default() -> Self {
        Self::Market
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Default, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PositionSide {
    #[default]
    Both,
    Long,
    Short,
}

#[derive(Eq, PartialEq, Debug, Serialize, Deserialize, Clone)]
pub enum TimeInForce {
    /// Good Till Canceled
    GTC,
    /// Immediate Or Cancel
    IOC,
    /// Fill or Kill
    FOK,
    /// Good till expired
    // GTX - Good Till Crossing (Post Only)
    // https://binance-docs.github.io/apidocs/futures/en/#public-endpoints-info
    GTX,
    #[serde(other)]
    Other,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct OrderRequest {
    pub symbol: String,
    pub side: OrderSide,
    pub position_side: Option<PositionSide>,
    #[serde(rename = "type")]
    pub order_type: OrderType,
    pub quantity: Option<f64>,
    pub price: Option<f64>,
    pub stop_price: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "timeInForce")]
    pub time_in_force: Option<TimeInForce>,
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GetOrderRequest {
    pub symbol: String,
    #[serde(rename = "orderId")]
    pub order_id: Option<String>,
    #[serde(rename = "origClientOrderId")]
    pub orig_client_order_id: Option<String>,
}

/// Status of an order, this can typically change over time
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OrderStatus {
    #[default]
    /// The order has been accepted by the engine.
    New,
    /// A part of the order has been filled.
    PartiallyFilled,
    /// The order has been completely filled.
    Filled,
    /// The order has been canceled by the user.
    Canceled,
    /// Currently unused
    PendingCancel,
    /// The order was not accepted by the engine and not processed.
    Rejected,
    /// The order was canceled according to the order type's rules (e.g. LIMIT FOK orders with no fill, LIMIT IOC or MARKET orders that partially fill) or by the exchange, (e.g. orders canceled during liquidation, orders canceled during maintenance)
    Expired,
    /// The order was canceled by the exchange due to STP trigger. (e.g. an order with EXPIRE_TAKER will match with existing orders on the book with the same account or same tradeGroupId)
    ExpiredInMatch,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WorkingType {
    MarkPrice,
    ContractPrice,
}

// Order: Object {"avgPrice": String("0.00"), "clientOrderId": String("IyJh35VJaDVZJFl2IJNGQP"), "closePosition": Bool(false),
//  "cumQty": String("0.000"), "cumQuote": String("0.00000"), "executedQty": String("0.000"), "goodTillDate": Number(0),
// "orderId": Number(4017658533), "origQty": String("0.020"), "origType": String("MARKET"), "positionSide": String("BOTH"),
//  "price": String("0.00"), "priceMatch": String("NONE"), "priceProtect": Bool(false), "reduceOnly": Bool(false),
// "selfTradePreventionMode": String("NONE"), "side": String("BUY"), "status": String("NEW"), "stopPrice": String("0.00"), "symbol": String("BTCUSDT"),
// "timeInForce": String("GTC"), "type": String("MARKET"), "updateTime": Number(1713713883047), "workingType": String("CONTRACT_PRICE")}

// Object {"avgPrice": String("65009.00000"), "clientOrderId": String("YWbv2arj3OFimAsZl0ESt3"), "closePosition": Bool(false),
// "cumQuote": String("1300.18000"), "executedQty": String("0.020"), "goodTillDate": Number(0), "orderId": Number(4017670481),
// "origQty": String("0.020"), "origType": String("MARKET"), "positionSide": String("BOTH"), "price": String("0.00"), "priceMatch": String("NONE"), "priceProtect": Bool(false), "reduceOnly": Bool(false), "selfTradePreventionMode": String("NONE"), "side": String("BUY"), "status": String("FILLED"), "stopPrice": String("0.00"), "symbol": String("BTCUSDT"), "time": Number(1713714869030), "timeInForce": String("GTC"), "type": String("MARKET"), "updateTime": Number(1713714869038), "workingType": String("CONTRACT_PRICE")}
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Transaction {
    pub client_order_id: String,
    #[serde(default)]
    #[serde(with = "string_or_float_opt")]
    pub cum_qty: Option<f64>,
    #[serde(with = "string_or_float")]
    pub cum_quote: f64,
    #[serde(with = "string_or_float")]
    pub executed_qty: f64,
    pub order_id: u64,
    #[serde(with = "string_or_float")]
    pub avg_price: f64,
    #[serde(with = "string_or_float")]
    pub orig_qty: f64,
    pub reduce_only: bool,
    pub side: OrderSide,
    pub position_side: PositionSide,
    pub status: OrderStatus,
    #[serde(with = "string_or_float")]
    pub stop_price: f64,
    pub close_position: bool,
    pub symbol: String,
    pub time_in_force: TimeInForce,
    #[serde(rename = "type")]
    pub type_name: OrderType,
    pub orig_type: OrderType,
    #[serde(default)]
    #[serde(with = "string_or_float_opt")]
    pub activate_price: Option<f64>,
    #[serde(default)]
    #[serde(with = "string_or_float_opt")]
    pub price_rate: Option<f64>,
    pub update_time: u64,
    pub working_type: WorkingType,
    price_protect: bool,
}

// Object {
//     "adlQuantile": Number(0),
//     "breakEvenPrice": String("0.0"),
//     "entryPrice": String("0.0"),
//     "isAutoAddMargin": String("false"),
//     "isolated": Bool(false),
//     "isolatedMargin": String("0.00000000"),
//     "isolatedWallet": String("0"),
//     "leverage": String("20"),
//     "liquidationPrice": String("0"),
//     "marginType": String("cross"),
//     "markPrice": String("0.00000000"),
//     "maxNotionalValue": String("25000"),
//     "notional": String("0"),
//     "positionAmt": String("0"),
//     "positionSide": String("BOTH"),
//     "symbol": String("XEMUSDT"),
//     "unRealizedProfit": String("0.00000000"),
//     "updateTime": Number(0)
// }

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BinanceFuturesPositionResponse {
    pub adl_quantile: f64,
    #[serde(with = "string_or_float")]
    pub break_even_price: f64,
    #[serde(with = "string_or_float")]
    pub entry_price: f64,
    #[serde(with = "string_or_bool")]
    pub is_auto_add_margin: bool,
    pub isolated: bool,
    #[serde(with = "string_or_float")]
    pub isolated_margin: f64,
    #[serde(with = "string_or_float")]
    pub isolated_wallet: f64,
    #[serde(with = "string_or_float")]
    pub leverage: f64,
    #[serde(with = "string_or_float")]
    pub liquidation_price: f64,
    pub margin_type: String,
    #[serde(with = "string_or_float")]
    pub mark_price: f64,
    #[serde(with = "string_or_float")]
    pub max_notional_value: f64,
    #[serde(with = "string_or_float")]
    pub notional: f64,
    #[serde(with = "string_or_float")]
    pub position_amt: f64,
    pub position_side: PositionSide,
    pub symbol: String,
    #[serde(with = "string_or_float", rename = "unRealizedProfit")]
    pub unrealized_profit: f64,
    pub update_time: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FuturesPosition {
    pub symbol: Symbol,
    pub quantity: f64,
    pub avg_price: Option<f64>,
    pub unrealized_profit: Option<f64>,
    pub position_side: Option<PositionSide>,
    pub leverage: Option<f64>,
    pub notional: Option<f64>,
    pub isolated: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BinanceFuturesChangeLeverageResponse {
    pub leverage: u8,
    #[serde(with = "string_or_float")]
    pub max_notional_value: f64,
    pub symbol: String,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct OrderCancellation {
    pub symbol: String,
    pub order_id: Option<u64>,
    pub orig_client_order_id: Option<String>,
    /// Used to uniquely identify this cancel. Automatically generated by default.
    pub new_client_order_id: Option<String>,
    /// Cannot be greater than 60000
    pub recv_window: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct OrderCanceled {
    pub symbol: String,
    pub orig_client_order_id: String,
    pub order_id: u64,
    pub client_order_id: String,
}

// [
//     1499040000000,      // Open time
//     "0.01634790",       // Open
//     "0.80000000",       // High
//     "0.01575800",       // Low
//     "0.01577100",       // Close
//     "148976.11427815",  // Volume
//     1499644799999,      // Close time
//     "2434.19055334",    // Quote asset volume
//     308,                // Number of trades
//     "1756.87402397",    // Taker buy base asset volume
//     "28.46694368",      // Taker buy quote asset volume
//     "17928899.62484339" // Ignore.
//   ]
#[derive(Debug, Clone, serde_tuple::Serialize_tuple, serde_tuple::Deserialize_tuple)]
#[serde(rename_all = "camelCase")]
pub struct BinanceFuturesKlineResponse {
    pub open_time: u64,
    #[serde(with = "string_or_float")]
    pub open: f64,
    #[serde(with = "string_or_float")]
    pub high: f64,
    #[serde(with = "string_or_float")]
    pub low: f64,
    #[serde(with = "string_or_float")]
    pub close: f64,
    #[serde(with = "string_or_float")]
    pub volume: f64,
    pub close_time: u64,
    #[serde(with = "string_or_float")]
    pub quote_asset_volume: f64,
    pub number_of_trades: u64,
    #[serde(with = "string_or_float")]
    pub taker_buy_base_asset_volume: f64,
    #[serde(with = "string_or_float")]
    pub taker_buy_quote_asset_volume: f64,
    #[serde(with = "string_or_float")]
    pub ignore: f64,
}
