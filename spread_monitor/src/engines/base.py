import threading
import time
import abc


class TradingBase(abc.ABC):
    def __init__(self):
        t = threading.Thread(target=self._loop)
        t.start()
    
    @abc.abstractclassmethod
    def place_order(self):
        pass

    def _loop(self):
        while(True):
            return