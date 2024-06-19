import ccxt.bingx


client = ccxt.bingx({
    'apiKey': "4P6YFvnHIDs3s2DLgJQgLmALKULCqjl34vVuBI0Db9gxuxXGhjtIWFTcoz0ekmFscZY2YUHIfH9l9xQM4Q",
    'secret': "xOr8V4NafpbhDovhi4rjDy6JpIpXRAE1ToaVYi7EPXnRDxyOfafA4v1MvdpvqmE9eJc4BqijpsAoHLjD8Fg",
    'type': 'swap',
    'enableRateLimit': True
})

# Get the balance
balance = client.fetch_balance()
# Get the maximum leverage
leverage = client.fetch_leverage(id='BTC-USDT')

print(balance)
