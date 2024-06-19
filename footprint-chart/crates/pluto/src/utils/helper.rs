use crate::*;

use std::sync::Arc;
use tokio::sync::Mutex;

pub fn start_binance_futures_websocket(channels: Vec<String>) -> PlutoEventReceiver {
    let (tx, rx) = tokio::sync::mpsc::channel(1024);
    tokio::spawn(async move {
        let mut binance_futures_websocket = BinanceFuturesWebsocket::new(Arc::new(Mutex::new(true)), |event| {
            match event {
                FuturesWebsocketEvent::Depth(e) => {
                    // let latancy = chrono::Utc::now().timestamp_millis() - e.event_time as i64;
                    // println!("Latancy: {} ms, symbol: {}, best bid price: {}, best ask price: {}", latancy, e.symbol, e.bids.first().unwrap().price, e.asks.first().unwrap().price);

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
                _ => {
                    // println!("Unknown event");
                }
            }

            Ok(())
        });
        binance_futures_websocket.start(channels).await.unwrap();
    });

    rx
}
