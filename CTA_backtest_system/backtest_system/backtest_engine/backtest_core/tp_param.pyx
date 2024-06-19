class TakeProfitParam:
    def __init__(self):
        pass
        
class StaticTakeProfitParam(TakeProfitParam):
    def __init__(self, price_layer_dict, is_long):
        # price : no of layer to cut loss
        super().__init__()
        self._price_layer_dict = price_layer_dict
        self._is_long = is_long
        self._next_tp_price = None
        self.update_next_tp_price()

    @property
    def next_tp_price(self):
        return self._next_tp_price

    def update_next_tp_price(self):
        if len(self._price_layer_dict) > 0:
            if self._is_long:
                self._next_tp_price = min(self._price_layer_dict.keys())
            else:
                self._next_tp_price = max(self._price_layer_dict.keys())
        else:
            self._next_tp_price = None

    def get_next_tp_layer(self):
        return self._price_layer_dict[self._next_tp_price]

    def hit_tp(self):
        self._price_layer_dict.pop(self._next_tp_price)
        self.update_next_tp_price()
