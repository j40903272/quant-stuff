import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_single_inst_position(
    np.ndarray[np.float64_t, ndim=1] np_bid, 
    np.ndarray[np.float64_t, ndim=1] np_ask, 
    np.ndarray[np.int_t, ndim=1] trade_signals,
    double no_of_layer,
    dict tp_sl_price_storage,
    str position_scale="base_value",
    double capital=10000.0,
    bint compound=False,
    double transaction_cost=0.0004
    ):

    cdef int data_len = len(trade_signals)
    cdef double current_position = 0.0
    cdef double current_layer = 0.0
    cdef double place_order = 0.0
    cdef double current_capital = capital

    cdef np.ndarray[np.float64_t, ndim=1] px = np.zeros(data_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] position = np.zeros(data_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] layers = np.zeros(data_len, dtype=np.float64)

    for idx in range(data_len):
        if current_position != 0.0:
            px[idx] = (np_bid[idx] + np_ask[idx]) / 2
        if place_order > 0.0:
            current_layer += place_order
            if position_scale == "proportion":
                current_position = current_capital * current_layer / no_of_layer / np_ask[idx]
            elif position_scale == "base_value":
                if current_layer == 0.0:
                    if compound:
                        current_capital -= current_position * np_ask[idx] * (1 - transaction_cost)
                    current_position = 0.0
                else:
                    if compound:
                        current_position += current_capital * place_order / no_of_layer / np_ask[idx] * (1 - transaction_cost)
                        current_capital -= current_capital * place_order / no_of_layer
                    else:
                        current_position += current_capital * place_order / no_of_layer / np_ask[idx]
                    
            else:
                current_position += place_order
            if idx - 1 in tp_sl_price_storage:
                px[idx] = tp_sl_price_storage[idx-1]
            else:
                px[idx] = np_ask[idx]  # sym1_bid
            place_order = 0.0
        elif place_order < 0.0:
            current_layer += place_order
            if position_scale == "proportion":
                current_position = current_capital * current_layer / no_of_layer / np_bid[idx]
            elif position_scale == "base_value":
                if current_layer == 0.0:
                    if compound:
                        current_capital += current_position * np_bid[idx] * (1 - transaction_cost)
                    current_position = 0.0
                else:
                    if compound:
                        current_capital += current_capital * place_order / no_of_layer
                        current_position += current_capital * place_order / no_of_layer / np_bid[idx] * (1 - transaction_cost)
                    else:
                        current_position += current_capital * place_order / no_of_layer / np_bid[idx]
            else:
                current_position += place_order
            if idx - 1 in tp_sl_price_storage:
                px[idx] = tp_sl_price_storage[idx-1]
            else:
                px[idx] = np_bid[idx]  # sym1_ask
            place_order = 0.0
        # cash[idx] = current_cash
        position[idx] = current_position

        if trade_signals[idx] != 0.0:
            place_order = trade_signals[idx]
        # position is made after the decision
        layers[idx] = current_layer
    
    return px, position, layers

@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_const_weight_portfolio_trade_position(
    np.ndarray np_bid,
    np.ndarray np_ask,
    np.ndarray[np.int_t, ndim=1] trade_signals,
    double no_of_layer,
    str position_scale="base_value",
    double capital=10000.0,
    weighting = None
    ):

    cdef int data_len = len(trade_signals)
    cdef np.ndarray[np.float64_t] current_position = np.zeros(len(weighting))
    cdef double current_layer = 0.0
    cdef double place_order = 0.0

    cdef np.ndarray[np.float64_t, ndim=1] n_weighting = np.array(weighting)

    cdef double sum_weight = sum(abs(i)for i in weighting)

    cdef np.ndarray[np.float64_t, ndim=2] px = np.zeros([data_len, len(weighting)], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] position = np.zeros([data_len, len(weighting)], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] layers = np.zeros([data_len, len(weighting)], dtype=np.float64)

    for idx in range(data_len):
        if np.absolute(current_position).sum() != 0.0:
            px[idx] = (np_bid[idx] + np_ask[idx]) / 2
        if place_order > 0.0:
            current_layer += place_order
            # for transaction cost calculation (long, short same fee)
            if position_scale == "proportion":
                for i, weight in enumerate(weighting):
                    a = capital * current_layer / no_of_layer / np_ask[idx]
                    if weight > 0 :
                        current_position[i] = weight / sum_weight * capital * current_layer / no_of_layer / np_ask[idx][i]
                    else:
                        current_position[i] = weight / sum_weight * capital * current_layer / no_of_layer / np_bid[idx][i]

            elif position_scale == "base_value":
                if current_layer == 0.0:
                    current_position = np.zeros(len(weighting))
                else:
                    base_amount = capital * place_order / no_of_layer / np.mean((np_ask[idx] + np_bid[idx])/2)
                    for i, weight in enumerate(weighting):
                        if weight > 0 :
                            current_position[i] += weight / sum_weight * base_amount
                        else:
                            current_position[i] += weight / sum_weight * base_amount
            else:
                current_position += place_order* n_weighting

            for i, weight in enumerate(weighting):
                if weight > 0 :
                    p = px[idx]
                    p[i] = np_ask[idx][i]
                else:
                    p = px[idx]
                    p[i] = np_bid[idx][i]

            place_order = 0.0
        elif place_order < 0.0:
            current_layer += place_order
            if position_scale == "proportion":
                for i, weight in enumerate(weighting):
                    if weight > 0 :
                        current_position[i] = weight / sum_weight * capital * current_layer / no_of_layer / np_bid[idx][i]
                    else:
                        current_position[i] = weight / sum_weight * capital * current_layer / no_of_layer / np_ask[idx][i]

            elif position_scale == "base_value":
                if current_layer == 0.0:
                    current_position = np.zeros(len(weighting))
                else:
                    base_amount = capital * place_order / no_of_layer / np.mean((np_ask[idx] + np_bid[idx])/2)
                    for i, weight in enumerate(weighting):
                        if weight > 0 :
                            current_position[i] += weight / sum_weight * base_amount
                        else:
                            current_position[i] += weight / sum_weight * base_amount
            else:
                current_position += place_order* n_weighting

            for i, weight in enumerate(weighting):
                if weight > 0:
                    p = px[idx]
                    p[i] = np_bid[idx][i]
                else:
                    p = px[idx]
                    p[i] = np_ask[idx][i]
            place_order = 0.0
        position[idx] = current_position

        if trade_signals[idx] != 0.0:
            place_order = trade_signals[idx]
        # position is made after the decision
        layers[idx] = current_layer

    return px, position, layers


@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_portfolio_trade_position(
    np.ndarray np_bid,
    np.ndarray np_ask,
    np.ndarray trade_signals,
    double no_of_layer,
    str position_scale="base_value",
    double capital=10000.0
    ):

    cdef int data_len = trade_signals.shape[1]
    cdef double current_position = 0.0
    cdef double current_layer = 0.0
    cdef double place_order = 0.0

    cdef np.ndarray[np.float64_t] px = np.zeros(data_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t] position = np.zeros(data_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] layers = np.zeros(data_len, dtype=np.float64)

    pass