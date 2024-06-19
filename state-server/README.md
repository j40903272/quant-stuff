# state-server
A HTTP server handle all strategy worker state.

## APIs

### get_orders(exchange, symbol, credential)

- orders: Array[Order]

### get_position(exchange, symbol, credential)

- Position

### get_orderbook(exchange, symbol, depth, credential)

- {"exchange":"Binance", "symbol":"ETH-USDT", "bids":[["99.9", "10.0"]], "asks":[["100.1", "10.0"]]}


## Class schema

### Order

    - exchange: "Binance"
    - symbol: "ETH-USDT-PERP"
    - type: "LIMIT"
    - positionSide: "LONG", "SHORT"
    - price: "1234.0"
    - qty: "10.0"
    - uuid: "xxxxx-xxxx-xxx-xxxxx"
    - orderId: "1273891273812"


### Position

- exchange: "Binance"
- symbol: "ETH-USDT-PERP"
- positionSide: "LONG", "SHORT"
- avgPrice: "1234.0"
- qty: "10.0"


flask --app main run