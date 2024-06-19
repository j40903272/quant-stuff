# tv-history

## Description

Download TradingView historical data.

## Usage

```bash
npm i
node tv-history.js
```

## Config

```js
// change this for start time
const stime = new Date() / 1000 - 86400 * 30 * 12 * 4 // 4 years
// change this for market symbol
const symbol = 'CME_MINI:ES1!'
```