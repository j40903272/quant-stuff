"""
USAGE:
ha = HA.run(
    data.get("Open"),
    data.get("High"),
    data.get("Low"),
    data.get("Close"),
    precision=precision,
)

precision is optional.
"""
from numba import njit
import numpy as np
import vectorbtpro as vbt


@njit(cache=True)
def ha_apply_func(open, high, low, close, precision=None):
    ha_open = np.full(close.shape, np.nan)
    ha_high = np.full(close.shape, np.nan)
    ha_low = np.full(close.shape, np.nan)

    ha_close = (open + high + low + close) / 4

    ha_open[0] = open[0]
    ha_high[0] = high[0]
    ha_low[0] = low[0]
    for i in range(ha_open.shape[0]):
        if i > 0:
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
            ha_high[i] = np.maximum(np.maximum(ha_open[i], ha_close[i]), high[i])
            ha_low[i] = np.minimum(np.minimum(ha_open[i], ha_close[i]), low[i])

    # Round to precision
    if precision is not None:
        for col in range(ha_close.shape[1]):
            np.round(ha_close[:, col], precision[col], ha_close[:, col])
            np.round(ha_open[:, col], precision[col], ha_open[:, col])
            np.round(ha_high[:, col], precision[col], ha_high[:, col])
            np.round(ha_low[:, col], precision[col], ha_low[:, col])

    return ha_open, ha_high, ha_low, ha_close


HA = vbt.IF(
    input_names=["open", "high", "low", "close"],
    output_names=["ha_open", "ha_high", "ha_low", "ha_close"],
).with_apply_func(
    ha_apply_func,
    kwargs_as_args=["precision"],
    precision=None,
)
