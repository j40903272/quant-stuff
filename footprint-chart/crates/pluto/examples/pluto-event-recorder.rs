// TZ=UTC RUST_LOG=pluto=debug cargo run --release --example pluto-event-recorder SUI USDC
use std::{io::Write, vec};

use pluto::*;

#[tokio::main]
async fn main() {
    let input_symbol = std::env::args().nth(1).unwrap_or("BTC".to_string());
    let input_currency = std::env::args().nth(2).unwrap_or("USDT".to_string());
    let symbol = Symbol::new("BINANCE-P", input_symbol.as_str(), Some(input_currency.as_str()));

    let ticker_channel = BinanceFuturesMarketParams::BookTicker(BookTickerParams::new(symbol.clone()));
    let trade_channel = BinanceFuturesMarketParams::Trade(TradeParams::new(symbol.clone()));
    let kline_channel = BinanceFuturesMarketParams::Kline(KlineParams::new(symbol.clone(), "1m".to_string()));
    let channels = vec![ticker_channel, trade_channel, kline_channel];
    let ws_subscription = EventSubscription::BinanceFuturesMarket(channels);

    let kline_restful = BinanceFuturesMarketRestfulParams::new(symbol.clone(), "1m".to_string(), None, Some(5000));
    let restful_subscription = EventSubscription::BinanceFuturesMarketRestful(kline_restful);

    let mut pluto_event_manager = PlutoEventManager::new();
    pluto_event_manager.subscribe(ws_subscription).unwrap();
    pluto_event_manager.subscribe(restful_subscription).unwrap();

    println!("Recording events for symbol: {}", symbol.to_string());

    let mut rx = pluto_event_manager.rx;
    let mut last_filename = String::new();
    let mut file: Option<std::fs::File> = None;
    loop {
        let event = rx.recv().await.unwrap();

        // create folder if not exists
        // "./outputs/pluto_event/{symbol}/yyyy/mm/dd"
        let foldername = format!("./outputs/pluto_event/{}/{}", symbol.to_string(), chrono::Local::now().format("%Y/%m/%d"));
        std::fs::create_dir_all(foldername).unwrap();

        // "./outputs/pluto_event/{symbol}/yyyy/mm/dd/hh.txt"
        let filename = format!("./outputs/pluto_event/{}/{}.txt", symbol.to_string(), chrono::Local::now().format("%Y/%m/%d/%H"));
        if file.is_none() || last_filename != filename {
            last_filename = filename.clone();
            file = Some(std::fs::File::create(filename).unwrap());
        }

        let s = format!("{}\n", serde_json::to_string(&event).unwrap());
        let _ = file.as_mut().unwrap().write_all(s.as_bytes());
    }
}
