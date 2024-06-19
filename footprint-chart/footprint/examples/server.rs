use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time;
use warp::ws::{Message, WebSocket};
use warp::Filter;

#[derive(Serialize, Deserialize)]
struct MyData {
    message: String,
    timestamp: u64,
}
use std::{io::Write, vec};

use pluto::*;

#[tokio::main]
async fn main() {
    let input_symbol = std::env::args().nth(1).unwrap_or("WLD".to_string());
    let input_currency = std::env::args().nth(2).unwrap_or("USDT".to_string());
    let symbol = Symbol::new("BINANCE-P", input_symbol.as_str(), Some(input_currency.as_str()));

    // let ticker_channel = BinanceFuturesMarketParams::BookTicker(BookTickerParams::new(symbol.clone()));
    let trade_channel = BinanceFuturesMarketParams::Trade(TradeParams::new(symbol.clone()));
    // let kline_channel = BinanceFuturesMarketParams::Kline(KlineParams::new(symbol.clone(), "1m".to_string()));
    let channels = vec![trade_channel];
    // let channels = vec![ticker_channel, trade_channel, kline_channel];
    let ws_subscription = EventSubscription::BinanceFuturesMarket(channels);

    // let kline_restful = BinanceFuturesMarketRestfulParams::new(symbol.clone(), "1m".to_string(), None, Some(5000));
    // let restful_subscription = EventSubscription::BinanceFuturesMarketRestful(kline_restful);

    let mut pluto_event_manager = PlutoEventManager::new();
    pluto_event_manager.subscribe(ws_subscription).unwrap();
    // pluto_event_manager.subscribe(restful_subscription).unwrap();

    let datas = Arc::new(Mutex::new(VecDeque::<PlutoEvent>::new()));

    let data_clone = datas.clone();

    let routes = warp::path("ws").and(warp::ws()).and(with_data(data_clone)).map(|ws: warp::ws::Ws, data| ws.on_upgrade(move |socket| user_connected(socket, data)));
    tokio::spawn(async move {
        println!("Server started at ws://127.0.0.1:3030");
        warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
    });


    // 開始更新資料
    let mut rx = pluto_event_manager.rx;
    loop {
        let event = rx.recv().await.unwrap();
        match event {
            PlutoEvent::TradeUpdate(_) => {
                let mut datas = datas.lock().await;
                datas.push_back(event);
            }
            _ => {}
        }
    }
}

fn with_data(data: Arc<Mutex<VecDeque<PlutoEvent>>>) -> impl Filter<Extract = (Arc<Mutex<VecDeque<PlutoEvent>>>,), Error = Infallible> + Clone {
    warp::any().map(move || data.clone())
}

async fn user_connected(ws: WebSocket, data: Arc<Mutex<VecDeque<PlutoEvent>>>) {
    let (mut tx, _rx) = ws.split();

    println!("User connected");
    loop {
        let json_data;
        {
            let mut datas = data.lock().await;
            // pop one
            let a = datas.pop_front();
            match a {
                Some(a) => {
                    json_data = serde_json::to_string(&a).unwrap();
                }
                None => {
                    json_data = "".to_string();
                }
            }
        }

        if json_data == "" {
            continue;
        }

        if tx.send(Message::text(json_data)).await.is_err() {
            break;
        }
    }
}
