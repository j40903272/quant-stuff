const TradingView = require('@mathieuc/tradingview')

function epochToISO(epoch) {
    return new Date(epoch * 1000).toISOString()
}

function getMarketData(market, timeframe, stime) {
    return new Promise((resolve, reject) => {
        const client = new TradingView.Client()
        const chart = new client.Session.Chart()
        chart.onUpdate(() => {
            resolve(chart.periods.map(({ time, open, max, min, close, volume }) => ({
                time, open, high: max, low: min, close, volume
            })))
        })
        chart.setMarket(market, { timeframe: '1H', range: -100, to: stime })
    })
}

async function main(market, stime) {
    let etime = new Date() / 1000
    let ans = []
    while (stime < etime) {
        let data = await getMarketData(market, '1H', stime)
        if (data.length === 1) break
        console.log(`Got ${data.length} candles from ${epochToISO(data[0].time)} to ${epochToISO(data[data.length - 1].time)}`)
        stime = data[0].time
        ans = data.concat(ans)
    }
    return ans.sort((a, b) => a.time - b.time)
}

async function tocsv(data) {
    let ans = []
    for (let i = 0; i < data.length; i++) {
        let d = data[i]
        ans.push(`${d.time},${d.open},${d.high},${d.low},${d.close},${d.volume}`)
    }
    return ans.join('\n')
}

const fs = require('fs')
const stime = new Date() / 1000 - 86400 * 30 * 12 * 4 // 4 years
// const symbol = 'TVC:NDQ'
// const symbol = 'SP:SPN'
const symbol = 'CME_MINI:ES1!'
main(stime, symbol).then(tocsv).then(csv => fs.writeFileSync(`tv-history-${symbol}.csv`, csv))