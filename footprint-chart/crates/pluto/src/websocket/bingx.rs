use crate::string_or_float;
use anyhow::Result;
use futures::SinkExt;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{from_value, Value};
use std::io::{Read, Write};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_tungstenite::connect_async;

use crate::Symbol;

pub enum SocketChannel {
    BookTicker(Symbol),
    Trade(Symbol),
    Kline(Symbol),
}

fn gzip_decompress(data: Vec<u8>) -> Vec<u8> {
    let mut decoder = flate2::read::GzDecoder::new(&data[..]);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed).unwrap();
    decompressed
}

fn gzip_compress(data: Vec<u8>) -> Vec<u8> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&data[..]).unwrap();
    encoder.finish().unwrap()
}

pub struct BingxWebsocket<'a> {
    on_off: Arc<Mutex<bool>>,
    handler: Box<dyn FnMut(BingxWebsocketEvent) -> Result<()> + 'a + Send>,
}

impl<'a> BingxWebsocket<'a> {
    pub fn new<Callback>(on_off: Arc<Mutex<bool>>, handler: Callback) -> BingxWebsocket<'a>
    where
        Callback: FnMut(BingxWebsocketEvent) -> Result<()> + 'a + Send,
    {
        BingxWebsocket { on_off: on_off, handler: Box::new(handler) }
    }

    fn get_endpoint(&self) -> String {
        "wss://open-api-swap.bingx.com/swap-market".to_string()
    }

    fn pluto_symbol_to_string(symbol: Symbol) -> String {
        format!("{}-{}", symbol.symbol, symbol.currency.unwrap_or_default())
    }

    fn channel_to_string(channel: SocketChannel) -> String {
        match channel {
            SocketChannel::BookTicker(symbol) => format!("{}@bookTicker", Self::pluto_symbol_to_string(symbol)),
            _ => todo!(),
        }
    }

    pub async fn start(&mut self, channels: Vec<SocketChannel>) -> Result<()> {
        let (ws_stream, _) = connect_async(self.get_endpoint()).await?;
        let (mut write, mut read) = ws_stream.split();

        for channel in channels {
            let payload = serde_json::json!({
                "reqType": "sub",
                "dataType": Self::channel_to_string(channel),
                "id": "1"
            });

            write.send(tokio_tungstenite::tungstenite::Message::Text(payload.to_string())).await?;
        }

        while *self.on_off.lock().await {
            if let Some(Ok(msg)) = read.next().await {
                match msg {
                    tokio_tungstenite::tungstenite::Message::Text(msg) => todo!(),
                    tokio_tungstenite::tungstenite::Message::Binary(data) => {
                        // gzip decompress
                        let decompressed = gzip_decompress(data);
                        let decompressed = String::from_utf8(decompressed).unwrap();

                        if decompressed == "Ping" {
                            write.send(tokio_tungstenite::tungstenite::Message::Text("Pong".to_string())).await?;
                            continue;
                        }

                        let value: Value = serde_json::from_str(&decompressed).unwrap();
                        if value["code"] != 0 || value["data"].is_null() {
                            continue;
                        }
                        let event: BingxWebsocketEvent = from_value(value["data"].clone()).unwrap();
                        (self.handler)(event)?;
                    }
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

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum BingxWebsocketEvent {
    BookTicker(BingxWebsocketBookTickerEvent),
}

// Object {"code": Number(0), "data": Object {"A": String("2.0826"), "B": String("1.7829"), "E": Number(1716240643168), "T": Number(1716240643154), "a": String("69227.5"), "b": String("69226.5"), "e": String("bookTicker"), "s": String("BTC-USDT"), "u": Number(836521754)}, "dataType": String("BTC-USDT@bookTicker")}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BingxWebsocketBookTickerEvent {
    #[serde(rename = "A")]
    #[serde(with = "string_or_float")]
    pub ask_qty: f64,
    #[serde(rename = "B")]
    #[serde(with = "string_or_float")]
    pub bid_qty: f64,
    #[serde(rename = "E")]
    pub event_time_ms: u64,
    #[serde(rename = "T")]
    pub trade_time_ms: u64,
    #[serde(rename = "a")]
    #[serde(with = "string_or_float")]
    pub ask_price: f64,
    #[serde(rename = "b")]
    #[serde(with = "string_or_float")]
    pub bid_price: f64,
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "u")]
    pub update_id: i64,
}
