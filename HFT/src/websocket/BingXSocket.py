import io
import gzip
import json
import uuid
import requests
import websocket
from .SocketBase import SocketBase

BINGX_SWAP_URL = "wss://open-api-swap.bingx.com/swap-market"


class BingXSocket(SocketBase):
    def __init__(self, channels=[]):
        super().__init__(BINGX_SWAP_URL)
        self.channels = channels

    def on_open(self, ws):
        id = str(uuid.uuid4())
        # subscribe all channels
        for channel in self.channels:
            ws.send(json.dumps({"id": id, "reqType": "sub", "dataType": channel}))

    def on_message(self, ws, message):
        compressed_data = gzip.GzipFile(fileobj=io.BytesIO(message), mode="rb")
        decompressed_data = compressed_data.read()
        utf8_data = decompressed_data.decode("utf-8")

        # ping pong, keep alive
        if utf8_data == "Ping":
            return ws.send("Pong")

        return super().on_message(ws, utf8_data)

