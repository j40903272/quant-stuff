import asyncio
import websockets
import json
import requests
import hmac
import base64
import zlib
import datetime
import time

class OKEX:
    
    def __init__(self, symbol):
        self.url = "wss://ws.okx.com:8443/ws/v5/public"
        

    def get_timestamp(self):
        now = datetime.datetime.now()
        t = now.isoformat("T", "milliseconds")
        return t + "Z"


    def get_server_time(self):
        url = "https://www.okx.com/api/v5/public/time"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['data'][0]['ts']
        else:
            return ""


    def get_local_timestamp(self):
        return int(time.time())


    def login_params(self, timestamp, api_key, passphrase, secret_key):
        message = timestamp + 'GET' + '/users/self/verify'

        mac = hmac.new(bytes(secret_key, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
        d = mac.digest()
        sign = base64.b64encode(d)

        login_param = {"op": "login", "args": [{"apiKey": api_key,
                                                "passphrase": passphrase,
                                                "timestamp": timestamp,
                                                "sign": sign.decode("utf-8")}]}
        login_str = json.dumps(login_param)
        return login_str


    def partial(self, res):
        data_obj = res['data'][0]
        bids = data_obj['bids']
        asks = data_obj['asks']
        instrument_id = res['arg']['instId']
        # print('全量数据bids为：' + str(bids))
        # print('档数为：' + str(len(bids)))
        # print('全量数据asks为：' + str(asks))
        # print('档数为：' + str(len(asks)))
        return bids, asks, instrument_id


    def update_bids(self, res, bids_p):
        # 获取增量bids数据
        bids_u = res['data'][0]['bids']
        # print('增量数据bids为：' + str(bids_u))
        # print('档数为：' + str(len(bids_u)))
        # bids合并
        for i in bids_u:
            bid_price = i[0]
            for j in bids_p:
                if bid_price == j[0]:
                    if i[1] == '0':
                        bids_p.remove(j)
                        break
                    else:
                        del j[1]
                        j.insert(1, i[1])
                        break
            else:
                if i[1] != "0":
                    bids_p.append(i)
        else:
            bids_p.sort(key=lambda price: self.sort_num(price[0]), reverse=True)
            # print('合并后的bids为：' + str(bids_p) + '，档数为：' + str(len(bids_p)))
        return bids_p


    def update_asks(self, res, asks_p):
        # 获取增量asks数据
        asks_u = res['data'][0]['asks']
        # print('增量数据asks为：' + str(asks_u))
        # print('档数为：' + str(len(asks_u)))
        # asks合并
        for i in asks_u:
            ask_price = i[0]
            for j in asks_p:
                if ask_price == j[0]:
                    if i[1] == '0':
                        asks_p.remove(j)
                        break
                    else:
                        del j[1]
                        j.insert(1, i[1])
                        break
            else:
                if i[1] != "0":
                    asks_p.append(i)
        else:
            asks_p.sort(key=lambda price: self.sort_num(price[0]))
            # print('合并后的asks为：' + str(asks_p) + '，档数为：' + str(len(asks_p)))
        return asks_p


    def sort_num(self, n):
        if n.isdigit():
            return int(n)
        else:
            return float(n)


    def check(self, bids, asks):
        # 获取bid档str
        bids_l = []
        bid_l = []
        count_bid = 1
        while count_bid <= 25:
            if count_bid > len(bids):
                break
            bids_l.append(bids[count_bid-1])
            count_bid += 1
        for j in bids_l:
            str_bid = ':'.join(j[0 : 2])
            bid_l.append(str_bid)
        # 获取ask档str
        asks_l = []
        ask_l = []
        count_ask = 1
        while count_ask <= 25:
            if count_ask > len(asks):
                break
            asks_l.append(asks[count_ask-1])
            count_ask += 1
        for k in asks_l:
            str_ask = ':'.join(k[0 : 2])
            ask_l.append(str_ask)
        # 拼接str
        num = ''
        if len(bid_l) == len(ask_l):
            for m in range(len(bid_l)):
                num += bid_l[m] + ':' + ask_l[m] + ':'
        elif len(bid_l) > len(ask_l):
            # bid档比ask档多
            for n in range(len(ask_l)):
                num += bid_l[n] + ':' + ask_l[n] + ':'
            for l in range(len(ask_l), len(bid_l)):
                num += bid_l[l] + ':'
        elif len(bid_l) < len(ask_l):
            # ask档比bid档多
            for n in range(len(bid_l)):
                num += bid_l[n] + ':' + ask_l[n] + ':'
            for l in range(len(bid_l), len(ask_l)):
                num += ask_l[l] + ':'

        new_num = num[:-1]
        int_checksum = zlib.crc32(new_num.encode())
        fina = self.change(int_checksum)
        return fina


    def change(self, num_old):
        num = pow(2, 31) - 1
        if num_old > num:
            out = num_old - num * 2 - 2
        else:
            out = num_old
        return out


    # subscribe channels un_need login
    async def subscribe_without_login(self):
        url = self.url
        channels = self.channels
        l = []
        while True:
            try:
                async with websockets.connect(url) as ws:
                    sub_param = {"op": "subscribe", "args": channels}
                    sub_str = json.dumps(sub_param)
                    await ws.send(sub_str)
                    print(f"send: {sub_str}")

                    while True:
                        try:
                            res = await asyncio.wait_for(ws.recv(), timeout=25)
                        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
                            try:
                                await ws.send('ping')
                                res = await ws.recv()
                                print(res)
                                continue
                            except Exception as e:
                                print("连接关闭，正在重连……")
                                break
                        # print(self.get_timestamp())
                        print(res)
                        res = eval(res)
                        if 'event' in res:
                            continue
                        for i in res['arg']:
                            if 'books' in res['arg'][i] and 'books5' not in res['arg'][i]:
                                # 订阅频道是深度频道
                                if res['action'] == 'snapshot':
                                    for m in l:
                                        if res['arg']['instId'] == m['instrument_id']:
                                            l.remove(m)
                                    # 获取首次全量深度数据
                                    bids_p, asks_p, instrument_id = self.partial(res)
                                    d = {}
                                    d['instrument_id'] = instrument_id
                                    d['bids_p'] = bids_p
                                    d['asks_p'] = asks_p
                                    l.append(d)

                                    # 校验checksum
                                    checksum = res['data'][0]['checksum']
                                    # print('推送数据的checksum为：' + str(checksum))
                                    check_num = self.check(bids_p, asks_p)
                                    # print('校验后的checksum为：' + str(check_num))
                                    if check_num == checksum:
                                        pass
                                        # print("校验结果为：True")
                                    else:
                                        pass
                                        # print("校验结果为：False，正在重新订阅……")

                                        # 取消订阅
                                        await self.unsubscribe_without_login(url, channels)
                                        # 发送订阅
                                        async with websockets.connect(url) as ws:
                                            sub_param = {"op": "subscribe", "args": channels}
                                            sub_str = json.dumps(sub_param)
                                            await ws.send(sub_str)
                                            print(f"send: {sub_str}")

                                elif res['action'] == 'update':
                                    for j in l:
                                        if res['arg']['instId'] == j['instrument_id']:
                                            # 获取全量数据
                                            bids_p = j['bids_p']
                                            asks_p = j['asks_p']
                                            # 获取合并后数据
                                            bids_p = self.update_bids(res, bids_p)
                                            asks_p = self.update_asks(res, asks_p)

                                            # 校验checksum
                                            checksum = res['data'][0]['checksum']
                                            # print('推送数据的checksum为：' + str(checksum))
                                            check_num = self.check(bids_p, asks_p)
                                            # print('校验后的checksum为：' + str(check_num))
                                            if check_num == checksum:
                                                print("校验结果为：True")
                                            else:
                                                print("校验结果为：False，正在重新订阅……")

                                                # 取消订阅
                                                await self.unsubscribe_without_login(url, channels)
                                                # 发送订阅
                                                async with websockets.connect(url) as ws:
                                                    sub_param = {"op": "subscribe", "args": channels}
                                                    sub_str = json.dumps(sub_param)
                                                    await ws.send(sub_str)
                                                    print(f"send: {sub_str}")
            except Exception as e:
                print("连接断开，正在重连……")
                continue


    # subscribe channels need login
    async def subscribe(self, url, api_key, passphrase, secret_key, channels):
        while True:
            try:
                async with websockets.connect(url) as ws:
                    # login
                    timestamp = str(self.get_local_timestamp())
                    login_str = self.login_params(timestamp, api_key, passphrase, secret_key)
                    await ws.send(login_str)
                    # print(f"send: {login_str}")
                    res = await ws.recv()
                    print(res)

                    # subscribe
                    sub_param = {"op": "subscribe", "args": channels}
                    sub_str = json.dumps(sub_param)
                    await ws.send(sub_str)
                    print(f"send: {sub_str}")

                    while True:
                        try:
                            res = await asyncio.wait_for(ws.recv(), timeout=25)
                        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
                            try:
                                await ws.send('ping')
                                res = await ws.recv()
                                print(res)
                                continue
                            except Exception as e:
                                print("连接关闭，正在重连……")
                                break

                        print(self.get_timestamp() + res)

            except Exception as e:
                print("连接断开，正在重连……")
                continue


    # trade
    async def trade(self, url, api_key, passphrase, secret_key, trade_param):
        while True:
            try:
                async with websockets.connect(url) as ws:
                    # login
                    timestamp = str(self.get_local_timestamp())
                    login_str = self.login_params(timestamp, api_key, passphrase, secret_key)
                    await ws.send(login_str)
                    # print(f"send: {login_str}")
                    res = await ws.recv()
                    print(res)

                    # trade
                    sub_str = json.dumps(trade_param)
                    await ws.send(sub_str)
                    print(f"send: {sub_str}")

                    while True:
                        try:
                            res = await asyncio.wait_for(ws.recv(), timeout=25)
                        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
                            try:
                                await ws.send('ping')
                                res = await ws.recv()
                                print(res)
                                continue
                            except Exception as e:
                                print("连接关闭，正在重连……")
                                break

                        print(self.get_timestamp() + res)

            except Exception as e:
                print("连接断开，正在重连……")
                continue


    # unsubscribe channels
    async def unsubscribe(self, url, api_key, passphrase, secret_key, channels):
        async with websockets.connect(url) as ws:
            # login
            timestamp = str(self.get_local_timestamp())
            login_str = self.login_params(timestamp, api_key, passphrase, secret_key)
            await ws.send(login_str)
            # print(f"send: {login_str}")

            res = await ws.recv()
            print(f"recv: {res}")

            # unsubscribe
            sub_param = {"op": "unsubscribe", "args": channels}
            sub_str = json.dumps(sub_param)
            await ws.send(sub_str)
            print(f"send: {sub_str}")

            res = await ws.recv()
            print(f"recv: {res}")


    # unsubscribe channels
    async def unsubscribe_without_login(self, url, channels):
        async with websockets.connect(url) as ws:
            # unsubscribe
            sub_param = {"op": "unsubscribe", "args": channels}
            sub_str = json.dumps(sub_param)
            await ws.send(sub_str)
            print(f"send: {sub_str}")

            res = await ws.recv()
            print(f"recv: {res}")

    def start_ws(self, channel):
        self.channels = [{"channel": channel, "instId": f"{self.symbol}-USD-SWAP"}]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.subscribe_without_login())

if __name__ == "__main__":
    api_key = ""
    secret_key = ""
    passphrase = ""

    url = "wss://ws.okx.com:8443/ws/v5/public"

    # 行情频道 tickers channel
    # channels = [{"channel": "tickers", "instId": "BTC-USD-210326"}]
    # 持仓总量频道 
    # channels = [{"channel": "open-interest", "instId": "BTC-USD-210326"}]
    # 交易频道
    # channels = [{"channel": "trades", "instId": "BTC-USD-201225"}]
    # 深度频道
    channels = [{"channel": "books", "instId": "BTC-USD-SWAP"}]

    okex = OKEX()

    loop = asyncio.get_event_loop()

    loop.run_until_complete(okex.subscribe_without_login())

    loop.close()