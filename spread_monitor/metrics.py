from prometheus_client import Counter, CollectorRegistry, Gauge, Summary, Histogram, start_http_server, generate_latest
from flask import Flask, Response

class DashBoard():
    def __init__(self):
        # Binance Gauges
        #self.binance_midprice_gauge = Gauge('binance_midprice_sd', 'Standard Deviation of Binance Mid Prices', ['symbol'])
        self.binance_spread_gauge = Gauge('binance_spread', 'Binance Spreads', ['symbol'])
        self.binance_spread_sd_gauge = Gauge('binance_spread_sd', 'Standard Deviation of Binance Spreads', ['symbol'])
        self.binance_midprice_sd_gauge = Gauge('binance_midprice_sd', 'Standard Deviation of Binance MidPrice', ['symbol'])

        # Bingx Gauges
        #self.bingx_midprice_gauge = Gauge('bingx_midprice_sd', 'Standard Deviation of Bingx Mid Prices', ['symbol'])
        self.bingx_spread_gauge = Gauge('bingx_spread', 'Bingx Spreads', ['symbol'])
        self.bingx_spread_sd_gauge = Gauge('bingx_spread_sd', 'Standard Deviation of Bingx Spreads', ['symbol'])
        self.bingx_midpric_sd_gauge = Gauge('bingx_midprice_sd', 'Standard Deviation of Bingx MidPrice', ['symbol'])


        # Differences Gauges
        self.midprice_cor_gauge = Gauge('midprice_cor', 'Correlation of Mid Price Differences between Binance and Bingx', ['symbol'])

        self.target_gauge = Gauge('target', 'Target', ['symbol'])

        self.registry = CollectorRegistry()
        self.registry.register(self.binance_spread_gauge)
        self.registry.register(self.binance_spread_sd_gauge)
        self.registry.register(self.binance_midprice_sd_gauge)

        self.registry.register(self.bingx_spread_gauge)
        self.registry.register(self.bingx_spread_sd_gauge)
        self.registry.register(self.bingx_midpric_sd_gauge)
        
        self.registry.register(self.midprice_cor_gauge)
        self.registry.register(self.target_gauge)
    
    def set_statistics(self, data):
        if data != {}:
            for symbol, value in data['volatility'].items():
               if 'Binance' in value:
                    # Update Binance metrics
                    self.binance_spread_gauge.labels(symbol=symbol).set(value['Binance']['spread'])
                    self.binance_spread_sd_gauge.labels(symbol=symbol).set(value['Binance']['spread_SD'])
                    self.binance_midprice_sd_gauge.labels(symbol=symbol).set(value['Binance']['midprice_SD'])
                    # Update Bingx metrics
                    self.bingx_spread_gauge.labels(symbol=symbol).set(value['Bingx']['spread'])
                    self.bingx_spread_sd_gauge.labels(symbol=symbol).set(value['Bingx']['spread_SD'])
                    self.bingx_midpric_sd_gauge.labels(symbol=symbol).set(value['Bingx']['midprice_SD'])
            for symbol, value in data['midprice_correlation'].items():
                if len(value) > 0:
                    self.midprice_cor_gauge.labels(symbol=symbol).set(value[-1])
    def set_target(self, data):
        if data != {}:
            for symbol, value in data:
                self.target_gauge.labels(symbol=symbol).set(value)


    def getRegistry(self):
        return generate_latest(self.registry)
