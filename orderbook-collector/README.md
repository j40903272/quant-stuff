# orderbook-collector

## Usage

### Docker

```bash
docker build -t orderbook-collector .
docker run --rm -v $(pwd)/data:/app/data -e SYMBOL="ETH-USDT" -e PLATFORM="binance" -e OUTPUT="/app/data" --name oc-eth-usdt-binance orderbook-collector
```