use std::sync::Arc;
use tokio::sync::Mutex;

use crate::BinanceExchange;
use crate::BinanceFuturesUserWebSocket;
use crate::BinanceFuturesWebsocket;
use crate::BingxWebsocket;
use crate::BingxWebsocketEvent;
use crate::FuturesWebsocketEvent;
use crate::OrderSide;
use crate::OrderStatus;
use crate::Orderbook;
use crate::OrderbookItem;
use crate::PlutoEvent;
use crate::PlutoEventReceiver;
use crate::PlutoEventSender;
use crate::SocketChannel;
use crate::Symbol;

pub struct DepthParams {
    pub symbol: Symbol,
    pub limit: u32,
}

impl DepthParams {
    pub fn new(symbol: Symbol, limit: u32) -> Self {
        Self { symbol, limit }
    }
}

pub struct TradeParams {
    pub symbol: Symbol,
}

impl TradeParams {
    pub fn new(symbol: Symbol) -> Self {
        Self { symbol }
    }
}

pub struct KlineParams {
    pub symbol: Symbol,
    pub interval: String,
}

impl KlineParams {
    pub fn new(symbol: Symbol, interval: String) -> Self {
        Self { symbol, interval }
    }
}

pub struct BookTickerParams {
    pub symbol: Symbol,
}

impl BookTickerParams {
    pub fn new(symbol: Symbol) -> Self {
        Self { symbol }
    }
}

pub struct BinanceFuturesMarketRestfulParams {
    pub symbol: Symbol,
    pub interval: String,
    pub limit: Option<u16>,
    pub cooldown_ms: Option<u32>,
}

impl BinanceFuturesMarketRestfulParams {
    pub fn new(symbol: Symbol, interval: String, limit: Option<u16>, cooldown_ms: Option<u32>) -> Self {
        Self { symbol, interval, limit, cooldown_ms }
    }
}

pub enum BinanceFuturesMarketParams {
    Depth(DepthParams),
    Trade(TradeParams),
    Kline(KlineParams),
    BookTicker(BookTickerParams),
}

pub struct BinanceFuturesUserParams {
    pub client: BinanceExchange,
}

pub enum BingxFuturesMarketParams {
    BookTicker(BookTickerParams),
}

pub enum EventSubscription {
    BinanceFuturesMarket(Vec<BinanceFuturesMarketParams>),
    BinanceFuturesUser(BinanceFuturesUserParams),
    BinanceFuturesMarketRestful(BinanceFuturesMarketRestfulParams),
    BingxFuturesMarket(Vec<BingxFuturesMarketParams>),
}

pub struct PlutoEventManager {
    tx: PlutoEventSender,
    pub rx: PlutoEventReceiver,
    // 暫時不用，之後會用來管理訂閱的內容
    subscriptions: Vec<EventSubscription>,
}

impl PlutoEventManager {
    pub fn new() -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(128);
        Self { tx, rx, subscriptions: Vec::new() }
    }

    pub fn subscribe(&mut self, subscription: EventSubscription) -> anyhow::Result<()> {
        match subscription {
            EventSubscription::BinanceFuturesMarket(params) => {
                start_binance_futures_market_stream(self.tx.clone(), params)?;
            }
            EventSubscription::BinanceFuturesMarketRestful(params) => {
                let tx_clone = self.tx.clone();
                let binance_exchange = BinanceExchange::new(crate::BinanceConfig::Futures, None, None);
                let symbol = params.symbol.clone();
                let interval = params.interval.clone();
                let limit = params.limit.unwrap_or(1000);
                let cooldown_ms = params.cooldown_ms.unwrap_or(5000);
                tokio::spawn(async move {
                    loop {
                        let kline = binance_exchange.get_klines(&symbol, interval.as_str(), Some(limit)).await.unwrap();
                        tx_clone.send(PlutoEvent::KlineUpdate(kline)).await.unwrap();
                        tokio::time::sleep(tokio::time::Duration::from_millis(cooldown_ms as u64)).await;
                    }
                });
            }
            EventSubscription::BinanceFuturesUser(params) => {
                start_binance_futures_user_stream(self.tx.clone(), params)?;
            }
            EventSubscription::BingxFuturesMarket(params) => {
                let channels = params
                    .iter()
                    .map(|param| match param {
                        BingxFuturesMarketParams::BookTicker(book_ticker_params) => SocketChannel::BookTicker(book_ticker_params.symbol.clone()),
                        _ => todo!(),
                    })
                    .collect::<Vec<SocketChannel>>();
                let tx_clone = self.tx.clone();
                tokio::spawn(async move {
                    let on_off = Arc::new(Mutex::new(true));

                    let mut ws = BingxWebsocket::new(on_off, |event: BingxWebsocketEvent| {
                        match event {
                            BingxWebsocketEvent::BookTicker(e) => {
                                let arr = e.symbol.split('-').collect::<Vec<&str>>();
                                let symbol = Symbol::new("BINGX-PS", arr[0], Some(arr[1]));
                                let orderbook = Orderbook {
                                    symbol: symbol,
                                    timestamp_ms: e.event_time_ms,
                                    bids: vec![OrderbookItem { price: e.bid_price, qty: e.bid_qty }],
                                    asks: vec![OrderbookItem { price: e.ask_price, qty: e.ask_qty }],
                                };
                                let tx_clone = tx_clone.clone();
                                tokio::spawn(async move {
                                    tx_clone.send(PlutoEvent::FullOrderbookUpdate(orderbook)).await.unwrap();
                                });
                            }
                            _ => {
                                todo!();
                            }
                        }
                        Ok(())
                    });
                    ws.start(channels).await.unwrap();
                });
            }
            _ => {
                return Err(anyhow::anyhow!("Subscription not supported"));
            }
        }
        Ok(())
    }
}

fn validate_binance_futures_market_params(params: &Vec<BinanceFuturesMarketParams>) -> anyhow::Result<()> {
    params.iter().try_for_each(|param| {
        match param {
            BinanceFuturesMarketParams::Depth(depth_params) => {
                if !["BINANCE-P", "BINANCE-F"].contains(&depth_params.symbol.exchange.as_str()) {
                    return Err(anyhow::anyhow!("Exchange only supports BINANCE-P, BINANCE-F"));
                }
                if ![5, 10, 20].contains(&depth_params.limit) {
                    return Err(anyhow::anyhow!("Limit must be 5, 10, 20"));
                }
            }
            BinanceFuturesMarketParams::Trade(parmas) => {
                if !["BINANCE-P", "BINANCE-F"].contains(&parmas.symbol.exchange.as_str()) {
                    return Err(anyhow::anyhow!("Exchange only supports BINANCE-P, BINANCE-F"));
                }
            }
            BinanceFuturesMarketParams::BookTicker(parmas) => {
                if !["BINANCE-P", "BINANCE-F"].contains(&parmas.symbol.exchange.as_str()) {
                    return Err(anyhow::anyhow!("Exchange only supports BINANCE-P, BINANCE-F"));
                }
            }
            BinanceFuturesMarketParams::Kline(parmas) => {
                if !["BINANCE-P", "BINANCE-F"].contains(&parmas.symbol.exchange.as_str()) {
                    return Err(anyhow::anyhow!("Exchange only supports BINANCE-P, BINANCE-F"));
                }
            }
            _ => {
                return Err(anyhow::anyhow!("Subscription not supported"));
            }
        }
        Ok(())
    })?;

    Ok(())
}

fn start_binance_futures_market_stream(tx: PlutoEventSender, params: Vec<BinanceFuturesMarketParams>) -> anyhow::Result<()> {
    validate_binance_futures_market_params(&params)?;
    let channels = params
        .iter()
        .map(|param| match param {
            BinanceFuturesMarketParams::Depth(depth_params) => {
                format!("{}@depth@100ms", BinanceExchange::pluto_symbol_to_binance(&depth_params.symbol).to_lowercase())
            }
            BinanceFuturesMarketParams::Trade(trade_params) => {
                format!("{}@trade", BinanceExchange::pluto_symbol_to_binance(&trade_params.symbol).to_lowercase())
            }
            BinanceFuturesMarketParams::BookTicker(book_ticker_params) => {
                format!("{}@bookTicker", BinanceExchange::pluto_symbol_to_binance(&book_ticker_params.symbol).to_lowercase())
            }
            BinanceFuturesMarketParams::Kline(kline_params) => {
                format!("{}@kline_{}", BinanceExchange::pluto_symbol_to_binance(&kline_params.symbol).to_lowercase(), kline_params.interval)
            }
            _ => {
                panic!("Subscription not supported");
            }
        })
        .collect::<Vec<String>>();

    tokio::spawn(async move {
        let mut ws = BinanceFuturesWebsocket::new(Arc::new(Mutex::new(true)), |event| {
            match event {
                FuturesWebsocketEvent::Depth(e) => {
                    let symbol = if e.symbol.ends_with("USDT") {
                        Symbol::new("BINANCE-P", &e.symbol[..e.symbol.len() - 4], Some("USDT"))
                    } else if e.symbol.ends_with("USDC") {
                        Symbol::new("BINANCE-P", &e.symbol[..e.symbol.len() - 4], Some("USDC"))
                    } else if e.symbol.ends_with("BTC") {
                        Symbol::new("BINANCE-P", &e.symbol[..e.symbol.len() - 3], Some("BTC"))
                    } else {
                        Symbol::new("BINANCE-P", &e.symbol, None)
                    };
                    let orderbook = Orderbook {
                        symbol: symbol,
                        timestamp_ms: e.event_time,
                        bids: e.bids.iter().map(|x| OrderbookItem { price: x.price, qty: x.qty }).collect(),
                        asks: e.asks.iter().map(|x| OrderbookItem { price: x.price, qty: x.qty }).collect(),
                    };
                    let tx_clone = tx.clone();
                    tokio::spawn(async move {
                        tx_clone.send(PlutoEvent::PartialOrderbookUpdate(orderbook)).await.unwrap();
                    });
                }
                FuturesWebsocketEvent::Trade(e) => {
                    let symbol = if e.symbol.ends_with("USDT") {
                        Symbol::new("BINANCE-P", &e.symbol[..e.symbol.len() - 4], Some("USDT"))
                    } else if e.symbol.ends_with("USDC") {
                        Symbol::new("BINANCE-P", &e.symbol[..e.symbol.len() - 4], Some("USDC"))
                    } else if e.symbol.ends_with("BTC") {
                        Symbol::new("BINANCE-P", &e.symbol[..e.symbol.len() - 3], Some("BTC"))
                    } else {
                        Symbol::new("BINANCE-P", &e.symbol, None)
                    };
                    let trade_data = crate::TradeData {
                        symbol: symbol,
                        timestamp_ms: e.trade_time,
                        price: e.price,
                        qty: e.qty,
                        is_buyer_maker: e.maker,
                    };
                    let tx_clone = tx.clone();
                    tokio::spawn(async move {
                        tx_clone.send(PlutoEvent::TradeUpdate(trade_data)).await.unwrap();
                    });
                }
                FuturesWebsocketEvent::BookTicker(e) => {
                    let symbol = if e.symbol.ends_with("USDT") {
                        Symbol::new("BINANCE-P", &e.symbol[..e.symbol.len() - 4], Some("USDT"))
                    } else if e.symbol.ends_with("USDC") {
                        Symbol::new("BINANCE-P", &e.symbol[..e.symbol.len() - 4], Some("USDC"))
                    } else if e.symbol.ends_with("BTC") {
                        Symbol::new("BINANCE-P", &e.symbol[..e.symbol.len() - 3], Some("BTC"))
                    } else {
                        Symbol::new("BINANCE-P", &e.symbol, None)
                    };
                    let orderbook = Orderbook {
                        symbol: symbol,
                        timestamp_ms: e.event_time,
                        bids: vec![OrderbookItem { price: e.best_bid_price, qty: e.best_bid_qty }],
                        asks: vec![OrderbookItem { price: e.best_ask_price, qty: e.best_ask_qty }],
                    };
                    let tx_clone = tx.clone();
                    tokio::spawn(async move {
                        tx_clone.send(PlutoEvent::FullOrderbookUpdate(orderbook)).await.unwrap();
                    });
                }
                FuturesWebsocketEvent::Kline(e) => {
                    let symbol = if e.symbol.ends_with("USDT") {
                        Symbol::new("BINANCE-P", &e.symbol[..e.symbol.len() - 4], Some("USDT"))
                    } else if e.symbol.ends_with("USDC") {
                        Symbol::new("BINANCE-P", &e.symbol[..e.symbol.len() - 4], Some("USDC"))
                    } else if e.symbol.ends_with("BTC") {
                        Symbol::new("BINANCE-P", &e.symbol[..e.symbol.len() - 3], Some("BTC"))
                    } else {
                        Symbol::new("BINANCE-P", &e.symbol, None)
                    };
                    let kline = crate::LastestKlineData {
                        symbol: symbol,
                        timestamp_ms: e.event_time,
                        open: e.kline.open,
                        high: e.kline.high,
                        low: e.kline.low,
                        close: e.kline.close,
                        volume: e.kline.base_asset_volume,
                        interval: Some(e.kline.interval),
                    };
                    let tx_clone = tx.clone();
                    tokio::spawn(async move {
                        tx_clone.send(PlutoEvent::LastestKlineUpdate(kline)).await.unwrap();
                    });
                }
                _ => {
                    println!("Unknown event");
                }
            }
            Ok(())
        });
        ws.start(channels).await.unwrap();
    });
    Ok(())
}

fn start_binance_futures_user_stream(tx: PlutoEventSender, params: BinanceFuturesUserParams) -> anyhow::Result<()> {
    let binance_exchange = params.client;
    tokio::spawn(async move {
        let listen_key = binance_exchange.get_userstream_listen_key().await.unwrap();
        let tx_clone = tx.clone();
        let mut ws = BinanceFuturesUserWebSocket::new(Arc::new(Mutex::new(true)), move |value| {
            if let Some(event_type) = value.get("e") {
                match event_type.as_str().unwrap() {
                    "ACCOUNT_UPDATE" => {
                        let event = value.get("a").unwrap();
                        println!("ACCOUNT_UPDATE: {}", event);
                    }
                    "ORDER_TRADE_UPDATE" => {
                        let event = value.get("o").unwrap();

                        let s = format!("{}", event.get("s").unwrap().as_str().unwrap());
                        let symbol = crate::binance_ticker_to_future_symbol(s);
                        // println!("symbol: {:?}", symbol);

                        let timestamp_ms = event.get("T").unwrap().as_u64().unwrap();
                        // println!("timestamp_ms: {}", timestamp_ms);
                        let price = event.get("p").unwrap().as_str().unwrap().parse::<f64>().unwrap();
                        // println!("price: {}", price);
                        let cum_qty = event.get("z").unwrap().as_str().unwrap().parse::<f64>().unwrap();
                        // println!("cum_qty: {}", cum_qty);
                        let qty = event.get("l").unwrap().as_str().unwrap().parse::<f64>().unwrap();
                        // println!("qty: {}", qty);
                        let side: OrderSide = serde_json::from_value(event.get("S").unwrap().clone()).unwrap();
                        // println!("side: {:?}", side);
                        let status: OrderStatus = serde_json::from_value(event.get("X").unwrap().clone()).unwrap();
                        // println!("status: {:?}", status);
                        let order_id = format!("{}", event.get("i").unwrap());
                        // println!("order_id: {}", order_id);
                        let order_type: crate::OrderType = serde_json::from_value(event.get("o").unwrap().clone()).unwrap();
                        // println!("order_type: {:?}", order_type);
                        let position_side: crate::PositionSide = serde_json::from_value(event.get("ps").unwrap().clone()).unwrap();
                        // println!("position_side: {:?}", position_side);
                        let time_in_force: crate::TimeInForce = serde_json::from_value(event.get("f").unwrap().clone()).unwrap();
                        // println!("time_in_force: {:?}", time_in_force);

                        let user_order_data = crate::UserOrderData {
                            symbol: symbol,
                            timestamp_ms: timestamp_ms,
                            price: price,
                            cum_qty: cum_qty,
                            qty: qty,
                            side: side,
                            status: status,
                            order_id: order_id,
                            order_type: order_type,
                            position_side: position_side,
                            time_in_force: Some(time_in_force),
                        };

                        let tx_clone = tx_clone.clone();
                        tokio::spawn(async move {
                            tx_clone.send(PlutoEvent::UserOrderUpdate(user_order_data)).await.unwrap();
                        });
                    }
                    _ => {
                        println!("Unknown event");
                    }
                }
            }
            Ok(())
        });

        // keep alive every 20 minutes
        let listen_key_clone = listen_key.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(25 * 60)).await;
                if let Err(e) = binance_exchange.keepalive_userstream_listen_key(&listen_key_clone).await {
                    println!("Error on keep alive: {}", e);
                }
            }
        });

        ws.start(listen_key).await.unwrap();
    });
    Ok(())
}
