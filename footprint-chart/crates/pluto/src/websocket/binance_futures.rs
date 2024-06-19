use anyhow::Result;
use futures::SinkExt;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{from_value, Value};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_tungstenite::connect_async;

use crate::websocket::model::BinanceFuturesDepthOrderBookEvent;
use crate::websocket::model::BinanceFuturesKlineEvent;
use crate::websocket::model::BinanceFuturesTradeEvent;
use crate::BinanceFuturesBookTickerEvent;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum FuturesWebsocketEvent {
    Depth(BinanceFuturesDepthOrderBookEvent),
    Trade(BinanceFuturesTradeEvent),
    BookTicker(BinanceFuturesBookTickerEvent),
    Kline(BinanceFuturesKlineEvent),
}

pub struct BinanceFuturesWebsocket<'a> {
    on_off: Arc<Mutex<bool>>,
    handler: Box<dyn FnMut(FuturesWebsocketEvent) -> Result<()> + 'a + Send>,
}

impl<'a> BinanceFuturesWebsocket<'a> {
    pub fn depth_channel(symbol: &str, levels: i32) -> String {
        match levels {
            5 => format!("{}@depth5@100ms", symbol.to_lowercase()),
            10 => format!("{}@depth10@100ms", symbol.to_lowercase()),
            20 => format!("{}@depth20@100ms", symbol.to_lowercase()),
            _ => panic!("Invalid levels"),
        }
    }

    pub fn new<Callback>(on_off: Arc<Mutex<bool>>, handler: Callback) -> BinanceFuturesWebsocket<'a>
    where
        Callback: FnMut(FuturesWebsocketEvent) -> Result<()> + 'a + Send,
    {
        BinanceFuturesWebsocket { on_off: on_off, handler: Box::new(handler) }
    }

    fn handle_msg(&mut self, msg: &str) -> Result<()> {
        let value: serde_json::Value = serde_json::from_str(msg)?;

        if let Ok(event) = serde_json::from_value::<FuturesWebsocketEvent>(value) {
            (self.handler)(event)?;
        } else {
            println!("Unknown event");
        }
        Ok(())
    }

    fn get_endpoint(&self) -> String {
        "wss://fstream.binance.com/ws".to_string()
    }

    pub async fn start(&mut self, channels: Vec<String>) -> Result<()> {
        let (ws_stream, _) = connect_async(self.get_endpoint()).await?;
        let (mut write, mut read) = ws_stream.split();

        let mut payload = serde_json::json!({
            "method": "SUBSCRIBE",
            "params": channels,
            "id": 1
        });

        write.send(tokio_tungstenite::tungstenite::Message::Text(payload.to_string())).await?;

        while *self.on_off.lock().await {
            if let Some(Ok(msg)) = read.next().await {
                match msg {
                    tokio_tungstenite::tungstenite::Message::Text(msg) => {
                        if let Err(e) = self.handle_msg(&msg) {
                            println!("{}", format!("Error on handling stream message: {}", e));
                        }
                    }
                    tokio_tungstenite::tungstenite::Message::Binary(_) => todo!(),
                    tokio_tungstenite::tungstenite::Message::Ping(payload) => {
                        write.send(tokio_tungstenite::tungstenite::Message::Pong(payload)).await?;
                    }
                    tokio_tungstenite::tungstenite::Message::Pong(_) => todo!(),
                    tokio_tungstenite::tungstenite::Message::Close(_) => todo!(),
                    tokio_tungstenite::tungstenite::Message::Frame(_) => todo!(),
                }
            }
        }

        Ok(())
    }
}

pub struct BinanceFuturesUserWebSocket<'a> {
    on_off: Arc<Mutex<bool>>,
    handler: Box<dyn FnMut(Value) -> Result<()> + 'a + Send>,
}

impl<'a> BinanceFuturesUserWebSocket<'a> {
    pub fn new<Callback>(on_off: Arc<Mutex<bool>>, handler: Callback) -> BinanceFuturesUserWebSocket<'a>
    where
        Callback: FnMut(Value) -> Result<()> + 'a + Send,
    {
        BinanceFuturesUserWebSocket { on_off: on_off, handler: Box::new(handler) }
    }

    fn get_endpoint(&self) -> String {
        "wss://fstream.binance.com/ws".to_string()
    }

    pub async fn start(&mut self, listen_key: String) -> Result<()> {
        let url = format!("{}/{}", self.get_endpoint(), listen_key);
        let (ws_stream, _) = connect_async(url).await?;
        let (mut write, mut read) = ws_stream.split();

        while *self.on_off.lock().await {
            if let Some(Ok(msg)) = read.next().await {
                match msg {
                    tokio_tungstenite::tungstenite::Message::Text(msg) => {
                        if let Ok(value) = serde_json::from_str(&msg) {
                            (self.handler)(value)?;
                        }
                    }
                    tokio_tungstenite::tungstenite::Message::Binary(_) => todo!(),
                    tokio_tungstenite::tungstenite::Message::Ping(payload) => {
                        write.send(tokio_tungstenite::tungstenite::Message::Pong(payload)).await?;
                    }
                    tokio_tungstenite::tungstenite::Message::Pong(_) => todo!(),
                    tokio_tungstenite::tungstenite::Message::Close(_) => todo!(),
                    tokio_tungstenite::tungstenite::Message::Frame(_) => todo!(),
                }
            }
        }

        Ok(())
    }
}
