const axios = require('axios');
const WebSocket = require('ws');
const crypto = require('crypto');
const express = require('express');
const expressWs = require('express-ws');
const ccxt = require('ccxt');

const app = express();
expressWs(app);

const BASE_URL = 'https://fapi.binance.com';
const API_KEY = 'API_KEY';
const API_SECRET = 'API_SECRET';

const exchange = new ccxt.binance({
    apiKey: 'API_KEY',
    secret: 'API_SECRET',
    options: {
        defaultType: 'future',
    }
});

let positions = {};
// Store subscribed clients
let clients = [];
const getSignature = () => {
    const queryString = `timestamp=${Date.now()}`;
    return crypto.createHmac('sha256', API_SECRET).update(queryString).digest('hex');
}
const fetchOpenPositions = async () => {
    const timestamp = Date.now();
    const signature = crypto.createHmac('sha256', API_SECRET).update(`timestamp=${timestamp}`).digest('hex');
    try {
        const response = await axios.get(`${BASE_URL}/fapi/v2/account`, {
            params: {
                timestamp: timestamp,
                signature: signature
            },
            headers: {
                'X-MBX-APIKEY': API_KEY
            }
        });
        const data = response.data;
        if (data.positions) {
            data.positions.forEach(position => {
                if (parseFloat(position.positionAmt) !== 0) {
                    positions[position.symbol] = {
                        symbol: position.symbol,
                        entryPrice: parseFloat(position.entryPrice),
                        amount: parseFloat(position.positionAmt),
                        pnl: parseFloat(position.unrealizedProfit),
                        ws: null
                    };
                }
            });
        }
    } catch (error) {
        console.error('Error fetching open positions:', error);
    }
};
const subscribeToTicker = (symbol) => {
    const coinWs = new WebSocket(`wss://fstream.binance.com/ws/${symbol.toLowerCase()}@ticker`);
    coinWs.on('message', coinData => {
        const position = positions[symbol];
        // Check if the position still exists
        if (!position) return;

        const marketData = JSON.parse(coinData);
        const currentPrice = parseFloat(marketData.c);
        const pnl = (currentPrice - position.entryPrice) * position.amount;

        // If PNL has changed, send updates to the users
        if (position.pnl !== pnl) {
            // Update the stored PNL for the position
            position.pnl = pnl;

            const openPositions = Object.values(positions).map(position => ({
                s: position.symbol,
                pa: position.amount,
                ep: position.entryPrice,
                pnl: position.pnl.toFixed(3)
            }));
            const message = JSON.stringify(openPositions);
            clients.forEach(client => client.send(message));
        }
    });
    // Store the coin's WebSocket connection
    positions[symbol].ws = coinWs;
};


// WebSocket endpoint for clients to subscribe to
app.ws('/subscribe', (ws, req) => {
    clients.push(ws);
    ws.on('close', () => {
        clients = clients.filter(client => client !== ws);
    });
    // For every open position, subscribe to its ticker's WebSocket and calculate PNL
    Object.keys(positions).forEach(symbol => {
        subscribeToTicker(symbol);
    });

    const openPositions = Object.values(positions).map(position => ({
        s: position.symbol,
        pa: position.amount,
        ep: position.entryPrice,
        pnl: position.pnl.toFixed(3)
    }));
    ws.send(JSON.stringify(openPositions));
});

axios.post(`${BASE_URL}/fapi/v1/listenKey`, `timestamp=${Date.now()}&signature=${getSignature()}`, {
    headers: {
        'X-MBX-APIKEY': API_KEY
    }
}).then(async response => {
    const listenKey = response.data.listenKey;
    await fetchOpenPositions();
    // Initiate WebSocket subscriptions for existing positions
    Object.values(positions).forEach(position => {
        const coinWs = new WebSocket(`wss://fstream.binance.com/ws/${position.s}@ticker`);
        coinWs.on('message', coinData => {
            // Check if the position still exists
            if (!positions[position.s]) return;

            const marketData = JSON.parse(coinData);
            const currentPrice = parseFloat(marketData.c);
            const pnl = (currentPrice - positions[position.s].entryPrice) * positions[position.s].amount;

            // If PNL has changed, send updates to the users
            if (positions[position.s].pnl !== pnl) {
                // Update the stored PNL for the position
                positions[position.s].pnl = pnl;

                clients.forEach(client => {
                    if (client.readyState === WebSocket.OPEN) {
                        client.send(JSON.stringify(positions));
                    }
                });
            }
        });
    });
    const ws = new WebSocket(`wss://fstream.binance.com/ws/${listenKey}`);

    ws.on('open', () => console.log('Connected to user data stream'));

    ws.on('message', data => {
        const eventData = JSON.parse(data);
        if (eventData.e === 'ACCOUNT_UPDATE') {
            eventData.a.P.forEach(position => {
                if (!positions[position.s] && position.pa !== '0') {
                    positions[position.s] = {
                        symbol: position.s,
                        entryPrice: parseFloat(position.ep),
                        amount: parseFloat(position.pa),
                        ws: null
                    };

                    // Subscribe to the coin's market data
                    const coinWs = new WebSocket(`wss://fstream.binance.com/ws/${position.s.toLowerCase()}@ticker`);
                    coinWs.on('message', coinData => {
                        // Check if the position still exists
                        if (!positions[position.s]) return;

                        const marketData = JSON.parse(coinData);
                        const currentPrice = parseFloat(marketData.c);
                        const pnl = (currentPrice - positions[position.s].entryPrice) * positions[position.s].amount;

                        // If PNL has changed, send updates to the users
                        if (positions[position.s].pnl !== pnl) {
                            // Update the stored PNL for the position
                            positions[position.s].pnl = pnl;
                            const openPositions = Object.values(positions).map(position => ({
                                s: position.symbol,
                                pa: position.amount,
                                ep: position.entryPrice,
                                pnl: position.pnl.toFixed(3)
                            }));
                            const message = JSON.stringify(openPositions);
                            clients.forEach(client => client.send(message));
                        }
                    });
                    // Store the coin's WebSocket connection
                    positions[position.s].ws = coinWs;
                } else if (positions[position.s] && position.pa === '0') {
                    if (positions[position.s].ws) {
                        // Close the coin's WebSocket connection
                        positions[position.s].ws.close();
                    }
                    // Remove the position from our tracked positions
                    delete positions[position.s];
                }
            });
            const openPositions = Object.values(positions).map(position => ({
                s: position.symbol,
                pa: position.amount,
                ep: position.entryPrice,
                pnl: position.pnl
            }));
            const message = JSON.stringify(openPositions);
            clients.forEach(client => client.send(message));
        }
    });
}).catch(err => console.error('Error obtaining listenKey:', err));

setInterval(() => {
    axios.put(`${BASE_URL}/fapi/v1/listenKey?listenKey=${listenKey}`, null, {
        headers: {
            'X-MBX-APIKEY': API_KEY
        }
    }).catch(err => console.error('Error updating listenKey:', err));
}, 30 * 60 * 1000);

async function tradingVolumes() {
    try {
        const endpointExchangeInfo = '/fapi/v1/exchangeInfo';
        const exchangeInfoUrl = `${BASE_URL}${endpointExchangeInfo}`;
        const exchangeInfoResponse = await axios.get(exchangeInfoUrl);
        const allFuturesSymbols = exchangeInfoResponse.data.symbols.map(symbolObj => symbolObj.symbol);

        const promises = allFuturesSymbols.map(async symbol => {
            const endpointUserTrades = '/fapi/v1/userTrades';
            const timestamp = Date.now();

            const queryString = `symbol=${symbol}&timestamp=${timestamp}`;
            const signature = crypto.createHmac('sha256', API_SECRET).update(queryString).digest('hex');

            const url = `${BASE_URL}${endpointUserTrades}?${queryString}&signature=${signature}`;
            const headers = {
                'X-MBX-APIKEY': API_KEY,
            };

            const response = await axios.get(url, {
                headers
            });

            let volumeForSymbol = 0;
            response.data.forEach(trade => {
                volumeForSymbol += parseFloat(trade.qty * trade.price);
            });

            return volumeForSymbol;
        });

        const volumes = await Promise.all(promises);
        const cumulativeVolume = volumes.reduce((acc, curr) => acc + curr, 0);

        return cumulativeVolume;

    } catch (error) {
        console.error('Error fetching futures trading volumes:', error);
        throw error;
    }
}
async function closePosition(symbol, positionAmt) {
    try {
        const side = parseFloat(positionAmt) > 0 ? 'SELL' : 'BUY';
        const amount = Math.abs(parseFloat(positionAmt));
        // Create a market order to close the position
        const order = await exchange.createOrder(symbol, 'market', side, amount);
    } catch (error) {
        console.error(`Error closing position for ${symbol}:`, error);
    }
}

async function closeAllOpenPositions() {
    try {
        const endpoint = '/fapi/v2/account';
        const timestamp = Date.now();

        const queryString = `timestamp=${timestamp}`;
        const signature = crypto.createHmac('sha256', API_SECRET).update(queryString).digest('hex');

        const url = `${BASE_URL}${endpoint}?${queryString}&signature=${signature}`;
        const headers = {
            'X-MBX-APIKEY': API_KEY,
        };

        const response = await axios.get(url, {
            headers
        });

        const open_positions = response.data.positions.filter(pos => parseFloat(pos.notional) !== 0)

        // Collect promises to close all positions
        const closePromises = open_positions.map(position => {
            if (position.positionAmt !== '0') {
                return closePosition(position.symbol, position.positionAmt);
            }
            return null;
        }).filter(p => p); // Filter out null promises

        // Wait for all positions to be closed concurrently
        await Promise.all(closePromises);

    } catch (error) {
        console.error('Error fetching or closing positions:', error);
    }
}
async function getAssets() {
    try {
        const endpoint = '/fapi/v2/account';
        const timestamp = Date.now();

        const queryString = `timestamp=${timestamp}`;
        const signature = crypto.createHmac('sha256', API_SECRET).update(queryString).digest('hex');

        const url = `${BASE_URL}${endpoint}?${queryString}&signature=${signature}`;
        const headers = {
            'X-MBX-APIKEY': API_KEY,
        };

        const response = await axios.get(url, {
            headers
        });
        return [response.data.totalMarginBalance, response.data.totalUnrealizedProfit, response.data.availableBalance];

    } catch (error) {
        console.error('Error fetching assets:', error);
        throw error;
    }
}
app.get('/tradingVolume', async (req, res) => {
    try {
        const data = await tradingVolumes();
        res.json(data);
    } catch (error) {
        res.status(500).json({
            message: 'Internal Server Error'
        });
    }
});
app.get('/assets', async (req, res) => {
    const [data1, data2, data3] = await getAssets();
    res.json([data1, data2, data3]);
});
app.post('/closeAllPositions', async (req, res) => {
    try {
        await closeAllOpenPositions();
        res.status(200).json({
            message: 'All open positions closed successfully!'
        });
    } catch (error) {
        console.error('Error closing positions:', error);
        res.status(500).json({
            message: 'Internal Server Error',
            details: error.message
        });
    }
});

const PORT = 4000;
app.listen(PORT, () => console.log(`Server started on ${PORT}`));
