class StaticStopLossParam:
    SL_MODE = 3

    def __init__(self, stop_loss_price):
        self._stop_loss_price = stop_loss_price

    @property
    def stop_loss_price(self):
        return self._stop_loss_price

    def get_stop_loss_price(self, trailing_stop_ref_price):
        return self._stop_loss_price


class AbsoluteTrailingStopLossParam:
    SL_MODE = 1

    def __init__(self, trailing_stop_value, trigger_price, is_long):
        self._trailing_stop_value = trailing_stop_value

        self._trigger_price = None
        if trigger_price is not None:
            self._trigger_price = trigger_price
            self._active_directly = False
        else:
            self._active_directly = True

        self._is_long = is_long

    @property
    def active_directly(self):
        return self._active_directly

    @property
    def trailing_stop_value(self):
        return self._trailing_stop_value

    @property
    def trigger_price(self):
        return self._trigger_price

    def get_stop_loss_price(self, trailing_stop_ref_price):
        if self._is_long:
            return trailing_stop_ref_price - self._trailing_stop_value
        else:
            return trailing_stop_ref_price + self._trailing_stop_value


class PercentageTrailingStopLossParam:
    SL_MODE = 2

    def __init__(self, stop_loss_percentage, trigger_price, is_long):
        self._stop_loss_percentage = stop_loss_percentage
        if trigger_price is not None:
            self._trigger_price = trigger_price
            self._active_directly = False
        else:
            self._active_directly = True
        self._is_long = is_long

    @property
    def stop_loss_percentage(self):
        return self._stop_loss_percentage

    @property
    def active_directly(self):
        return self._active_directly

    @property
    def trigger_price(self):
        return self._trigger_price

    def get_stop_loss_price(self, trailing_stop_ref_price):
        if self._is_long:
            return trailing_stop_ref_price * (1 - self._stop_loss_percentage)
        else:
            return trailing_stop_ref_price * (1 + self._stop_loss_percentage)
