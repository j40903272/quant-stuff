from numba import njit
import numpy as np
import vectorbtpro as vbt
from vectorbtpro import _typing as tp
from vectorbtpro.generic import nb as generic_nb, enums as generic_enums
from numba import prange

@njit(cache=True, nogil=True)
def adx_1d_nb(high: tp.Array1d, low: tp.Array1d, close: tp.Array1d, window: int = 14, wtype: int = 3) -> tp.Array1d:
    dm_pos=np.full(high.shape, np.nan)
    dm_neg=np.full(high.shape, np.nan)
    tr,atr=vbt.indicators.nb.atr_nb(high=np.expand_dims(high, axis=1), low=np.expand_dims(low, axis=1), close=np.expand_dims(close, axis=1), window=window, wtype=wtype, adjust=False, minp=None, tr=None, cache_dict=None)
    for i in range(1, high.shape[0]):
        dm_pos[i]=high[i]-high[i-1] 
        dm_neg[i]=low[i-1]-low[i]
        if dm_pos[i]>dm_neg[i]:
            dm_neg[i]=0
            if dm_pos[i]<0:
                dm_pos[i]=0
        elif dm_pos[i]<dm_neg[i]: 
            dm_pos[i]=0
            if dm_neg[i]<0:
                dm_neg[i]=0
    smoothed_dm_pos=vbt.generic.nb.ma_1d_nb(arr=dm_pos, window=window, wtype=wtype, minp=None, adjust=False)
    smoothed_dm_neg=vbt.generic.nb.ma_1d_nb(arr=dm_neg, window=window, wtype=wtype, minp=None, adjust=False)
    di_pos=smoothed_dm_pos/atr[:, 0]*100
    di_neg=smoothed_dm_neg/atr[:, 0]*100
    dx=np.abs((di_pos-di_neg))/np.abs((di_pos+di_neg))*100
    adx=vbt.generic.nb.ma_1d_nb(arr=dx, window=window, wtype=wtype, minp=None, adjust=False)
    return adx

@njit(cache=True, nogil=True)
def adx_nb(high: tp.Array2d, low: tp.Array2d, close: tp.Array2d, window: int = 14, wtype: int = 3) -> tp.Array2d:
    adx = np.empty_like(high, dtype=np.float_)
    for col in prange(high.shape[1]):
        adx[:, col] = adx_1d_nb(high[:, col], low[:, col], close[:, col], window, wtype)
    return adx

ADX = vbt.IF(
    input_names=["high", "low", "close"],
    param_names=["window", "wtype"],
    output_names=["adx"],
).with_apply_func(
    adx_nb,
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=14,
    wtype="wilder",
)
