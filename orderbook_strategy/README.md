## 安裝套進
```
pip install -r ./requirements.txt
```
## 執行方式
- 修改config.ini
```
[DEFUALT]
symbols = THETA-USDT,UNI-USDT,OP-USDT,FET-USDT ;要執行的幣種
threshold = 0.00055;取波動前0.0001的資料當進場
```
- 執行抓orderbook資料的程式 main.py
- 執行自動回測的程式 auto_backtesting.py

       