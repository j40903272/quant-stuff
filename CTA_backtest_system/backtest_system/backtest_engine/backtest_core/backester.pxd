cimport numpy as np

cdef class BackTester(objects):
    cdef np.ndarray _trade_signals
    cdef np.ndarray[np.int_t, ndim=1] _made_action
    cdef int _current_position
    cdef double avg_pf_value
    cdef np.ndarray avg_pf_value_record
    cdef int _buffer_period_cnt
    cdef int _buffer_period
    cdef int _can_trade_flag
    cdef int layer_width
    cdef int _layer
    cdef int _no_of_symbols
    cdef str _portfolio_trade_method

    cdef double _trailing_stop_thres
    cdef double _trailing_stop_ref_px
    cdef int _SL_mode
    cdef double _take_profit_px

    cdef list target_fields
    cdef np.ndarray data
    cdef np.ndarray _dt_index
    cdef int _idx
    cdef int _trade_ref_price
    cdef np.ndarray _row
    cdef dict _search_dict

    cdef str _data_path
    cdef str _name