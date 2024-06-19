import io
import gzip
import json
import uuid
import requests
import websocket


class SocketBase(object):
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.close_reason = ""

    def handler(self, message):
        pass

    def on_open(self, ws):
        pass

    def on_message(self, ws, message):
        self.handler(message)

    def on_error(self, ws, error):
        print("on_error", error)

    def on_close(self, ws, close_status_code, close_msg):
        self.close_reason = close_msg if close_msg else "Unknown"

    def run(self):
        self.ws = websocket.WebSocketApp(
            self.url, None, self.on_open, self.on_message, self.on_error, self.on_close
        )
        self.ws.run_forever()

    def start(self, auto_reconnect=True):
        while True:
            try:
                self.run()
            except Exception as e:
                print("Error", e)
            if not auto_reconnect or self.close_reason == "Unknown":
                break
