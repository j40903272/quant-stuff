symbols="BTC-USDT,ETH-USDT"

for symbol in $(echo $symbols | sed "s/,/ /g")
do
    docker run -d --restart=always -v $(pwd)/data:/app/data -e SYMBOL="$symbol" -e PLATFORM="binance" -e OUTPUT="/app/data" --name "oc-binance-$symbol" orderbook-collector
    sleep 3
    docker run -d --restart=always -v $(pwd)/data:/app/data -e SYMBOL="$symbol" -e PLATFORM="bingx" -e OUTPUT="/app/data" --name "oc-bingx-$symbol" orderbook-collector
    sleep 3
done
