from metrics import DashBoard
from refractory import MarketMonitor
from refractory import fetch_common_usdt_symbols
from flask import Flask, Response

app = Flask(__name__)

try:
    symbols = fetch_common_usdt_symbols()
except Exception as e:
    print(str(e))


monitor = MarketMonitor(symbols)
dashboard = DashBoard()

# Helper function to calculate weighted sums
def calculate_weighted_sums(ranked_items, weighted_sums):
    for idx, (symbol, data) in enumerate(ranked_items, start=1):
        weighted_sums[symbol] = weighted_sums.get(symbol, 0) + (len(ranked_items) - idx + 1)


@app.get('/metrics')
def metrics():
    def calculate_weighted_sums(ranked_items, weighted_sums, weight_factor):
        for idx, (symbol, data) in enumerate(ranked_items, start=1):
            weighted_sums[symbol] = weighted_sums.get(symbol, 0) + weight_factor * (MAX_RANK - idx + 1)

    # Filter out entries with empty lists
    volatility_items = [(symbol, data) for symbol, data in monitor.statistics.get('volatility', {}).items() if data]
    midprice_cor_items = [(symbol, data[-1]) for symbol, data in monitor.statistics.get('midprice_correlation', {}).items() if data]


    MAX_RANK = len(volatility_items)

    # Sort
    binance_spread_ranked = sorted(volatility_items, key=lambda x: x[1]['Binance']['spread'], reverse=True)[:MAX_RANK]
    binance_spread_SD_ranked = sorted(volatility_items, key=lambda x: x[1]['Binance']['spread_SD'], reverse=True)[:MAX_RANK]
    binance_midprice_SD_ranked = sorted(volatility_items, key=lambda x: x[1]['Binance']['midprice_SD'], reverse=True)[:MAX_RANK]

    bingx_spread_ranked = sorted(volatility_items, key=lambda x: x[1]['Bingx']['spread'], reverse=True)[:MAX_RANK]
    bingx_spread_SD_ranked = sorted(volatility_items, key=lambda x: x[1]['Bingx']['spread_SD'], reverse=True)[:MAX_RANK]
    bingx_midprice_SD_ranked = sorted(volatility_items, key=lambda x: x[1]['Bingx']['midprice_SD'], reverse=True)[:MAX_RANK]

    midprice_cor_ranked = sorted(midprice_cor_items, key=lambda x: x[1], reverse=False)[:MAX_RANK]

    weighted_sums = {}

    # Calculate weighted sums
    calculate_weighted_sums(binance_spread_ranked, weighted_sums, 1)
    calculate_weighted_sums(binance_spread_SD_ranked, weighted_sums, 1)
    calculate_weighted_sums(binance_midprice_SD_ranked, weighted_sums, 2)
    calculate_weighted_sums(bingx_spread_ranked, weighted_sums, 1)
    calculate_weighted_sums(bingx_spread_SD_ranked, weighted_sums, 1)
    calculate_weighted_sums(bingx_midprice_SD_ranked, weighted_sums, 2)
    calculate_weighted_sums(midprice_cor_ranked, weighted_sums, 1)

    # Filter and sort the results
    weight_items = [(symbol, data) for symbol, data in weighted_sums.items() if data]
    weight_ranked = sorted(weight_items, key=lambda x: x[1], reverse=True)[:MAX_RANK]

    dashboard.set_statistics(monitor.statistics)
    dashboard.set_target(weight_ranked)
    
    # #Print or use the results as needed with symbol and its value
    # print("Binance spread top 50 symbols:")
    # for symbol, data in binance_spread_ranked:
    #     print(f"{symbol}: {data['Binance']['spread']}")

    # print("\nBinance spread_SD top 50 symbols:")
    # for symbol, data in binance_spread_SD_ranked:
    #     print(f"{symbol}: {data['Binance']['spread_SD']}")

    # print("\nBingx spread top 50 symbols:")
    # for symbol, data in bingx_spread_ranked:
    #     print(f"{symbol}: {data['Bingx']['spread']}")

    # print("\nBingx spread_SD top 50 symbols:")
    # for symbol, data in bingx_spread_SD_ranked:
    #     print(f"{symbol}: {data['Bingx']['spread_SD']}")

    # print("\nPrice Correlation top 50 symbols:")
    # for symbol, data in midprice_cor_ranked:
    #     print(f"{symbol}: {data}")

    # print("\weight_ranked:")
    # for symbol, data in weight_ranked:
    #     print(f"{symbol}: {data}")

    return Response(dashboard.getRegistry(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=8000, debug= False)