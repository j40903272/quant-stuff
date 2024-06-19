import numpy as np
import pandas as pd
from datetime import datetime

import ibapi
from ibapi.common import TickerId, BarData
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.utils import iswrapper

class APP(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self,self)
        EWrapper.__init__(self)
    def error(self,reqId,errorCode,errorString):
        print("Error: ",reqId, " ",errorCode," ",errorString)
    def historicalData(self, regId, bar):
        print("HistoricalData: ",reqId,"Date:",bar.date,"Open:",bar.open,"High:",bar.high,"Low:",bar.low,"Close",bar.close,
        "Volume:",bar.volume,"Count",bar.count)

app = APP()
app.connect("127.0.0.1",7496,0)

contract = Contract()
contract.symbol = "AMZN"
contract.secType = "STK"
contract.currency = "USD"
contract.exchange = "SMART"

current = datetime.now().strftime("%Y%m%d %H:%M:%S")

app.reqHistoricalData(1,contract,current,"1 D","1 min","TRADES",0, 1, False, [])
app.run()

