import os
from datetime import datetime

import vectorbtpro as vbt

from tests.utils import *

close_ts = pd.DataFrame(
    {"a": [1, 2, 1, 2, 3, 2], "b": [3, 2, 3, 2, 1, 2]},
    index=pd.Index(
        [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
            datetime(2020, 1, 5),
            datetime(2020, 1, 6),
        ]
    ),
)

up_ths = [np.array([[1, 1 / 2]]), np.array([[2, 1 / 2]]), np.array([[3, 1 / 2]])]
down_ths = [np.array([[1 / 2, 1 / 3]]), np.array([[1 / 2, 2 / 3]]), np.array([[1 / 2, 3 / 4]])]


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


# ############# generators ############# #


class TestGenerators:
    def test_FMEAN(self):
        assert_frame_equal(
            vbt.FMEAN.run(close_ts, window=(2, 3), wtype="simple").fmean,
            pd.DataFrame(
                np.array(
                    [
                        [1.5, 2.5, 1.6666666666666667, 2.3333333333333335],
                        [1.5, 2.5, 2.0, 2.0],
                        [2.5, 1.5, 2.3333333333333335, 1.6666666666666667],
                        [2.5, 1.5, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (2, "simple", "a"),
                        (2, "simple", "b"),
                        (3, "simple", "a"),
                        (3, "simple", "b"),
                    ],
                    names=["fmean_window", "fmean_wtype", None],
                ),
            ),
        )
        assert_frame_equal(
            vbt.FMEAN.run(close_ts, window=(2, 3), wtype="exp").fmean,
            pd.DataFrame(
                np.array(
                    [
                        [1.8024691358024691, 2.197530864197531, 1.8125, 2.1875],
                        [1.4074074074074074, 2.5925925925925926, 1.625, 2.375],
                        [2.2222222222222223, 1.7777777777777777, 2.25, 1.75],
                        [2.666666666666667, 1.3333333333333335, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (2, "exp", "a"),
                        (2, "exp", "b"),
                        (3, "exp", "a"),
                        (3, "exp", "b"),
                    ],
                    names=["fmean_window", "fmean_wtype", None],
                ),
            ),
        )

    def test_FSTD(self):
        assert_frame_equal(
            vbt.FSTD.run(close_ts, window=(2, 3), wtype="simple").fstd,
            pd.DataFrame(
                np.array(
                    [
                        [0.5, 0.5, 0.4714045207910384, 0.4714045207910183],
                        [0.5, 0.5, 0.816496580927726, 0.816496580927726],
                        [0.5, 0.5, 0.4714045207910183, 0.4714045207910384],
                        [0.5, 0.5, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (2, "simple", "a"),
                        (2, "simple", "b"),
                        (3, "simple", "a"),
                        (3, "simple", "b"),
                    ],
                    names=["fstd_window", "fstd_wtype", None],
                ),
            ),
        )
        assert_frame_equal(
            vbt.FSTD.run(close_ts, window=(2, 3), wtype="exp").fstd,
            pd.DataFrame(
                np.array(
                    [
                        [0.64486716348143, 0.6448671634814303, 0.6462561866810479, 0.6462561866810479],
                        [0.8833005039168617, 0.8833005039168604, 0.8591246929842246, 0.8591246929842246],
                        [0.5916079783099623, 0.5916079783099623, 0.5477225575051662, 0.5477225575051662],
                        [0.7071067811865476, 0.7071067811865476, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (2, "exp", "a"),
                        (2, "exp", "b"),
                        (3, "exp", "a"),
                        (3, "exp", "b"),
                    ],
                    names=["fstd_window", "fstd_wtype", None],
                ),
            ),
        )

    def test_FMIN(self):
        assert_frame_equal(
            vbt.FMIN.run(close_ts, window=(2, 3)).fmin,
            pd.DataFrame(
                np.array(
                    [
                        [1.0, 2.0, 1.0, 2.0],
                        [1.0, 2.0, 1.0, 1.0],
                        [2.0, 1.0, 2.0, 1.0],
                        [2.0, 1.0, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (2, "a"),
                        (2, "b"),
                        (3, "a"),
                        (3, "b"),
                    ],
                    names=["fmin_window", None],
                ),
            ),
        )

    def test_FMAX(self):
        assert_frame_equal(
            vbt.FMAX.run(close_ts, window=(2, 3)).fmax,
            pd.DataFrame(
                np.array(
                    [
                        [2.0, 3.0, 2.0, 3.0],
                        [2.0, 3.0, 3.0, 3.0],
                        [3.0, 2.0, 3.0, 2.0],
                        [3.0, 2.0, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (2, "a"),
                        (2, "b"),
                        (3, "a"),
                        (3, "b"),
                    ],
                    names=["fmax_window", None],
                ),
            ),
        )

    def test_FIXLB(self):
        assert_frame_equal(
            vbt.FIXLB.run(close_ts, n=(2, 3)).labels,
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 1.0, -0.3333333333333333],
                        [0.0, 0.0, 0.5, -0.5],
                        [2.0, -0.6666666666666666, 1.0, -0.3333333333333333],
                        [0.0, 0.0, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (2, "a"),
                        (2, "b"),
                        (3, "a"),
                        (3, "b"),
                    ],
                    names=["fixlb_n", None],
                ),
            ),
        )

    def test_MEANLB(self):
        assert_frame_equal(
            vbt.MEANLB.run(close_ts, window=(2, 3), wtype="simple").labels,
            pd.DataFrame(
                np.array(
                    [
                        [0.5, -0.16666666666666666, 0.6666666666666667, -0.22222222222222218],
                        [-0.25, 0.25, 0.0, 0.0],
                        [1.5, -0.5, 1.3333333333333335, -0.4444444444444444],
                        [0.25, -0.25, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (2, "simple", "a"),
                        (2, "simple", "b"),
                        (3, "simple", "a"),
                        (3, "simple", "b"),
                    ],
                    names=["meanlb_window", "meanlb_wtype", None],
                ),
            ),
        )
        assert_frame_equal(
            vbt.MEANLB.run(close_ts, window=(2, 3), wtype="exp").labels,
            pd.DataFrame(
                np.array(
                    [
                        [0.8024691358024691, -0.2674897119341564, 0.8125, -0.2708333333333333],
                        [-0.2962962962962963, 0.2962962962962963, -0.1875, 0.1875],
                        [1.2222222222222223, -0.40740740740740744, 1.25, -0.4166666666666667],
                        [0.3333333333333335, -0.33333333333333326, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (2, "exp", "a"),
                        (2, "exp", "b"),
                        (3, "exp", "a"),
                        (3, "exp", "b"),
                    ],
                    names=["meanlb_window", "meanlb_wtype", None],
                ),
            ),
        )

    def test_LEXLB(self):
        assert_frame_equal(
            vbt.LEXLB.run(close_ts, up_th=up_ths, down_th=down_ths).labels,
            pd.DataFrame(
                np.array(
                    [
                        [-1, 1, -1, 1, 0, 0],
                        [1, -1, 0, 0, 0, 0],
                        [-1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [1, -1, 1, -1, 0, 0],
                        [0, 1, 0, 1, 0, 0],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("array_0", "array_0", "a"),
                        ("array_0", "array_0", "b"),
                        ("array_1", "array_1", "a"),
                        ("array_1", "array_1", "b"),
                        ("array_2", "array_2", "a"),
                        ("array_2", "array_2", "b"),
                    ],
                    names=["lexlb_up_th", "lexlb_down_th", None],
                ),
            ),
        )

    def test_TRENDLB(self):
        assert_frame_equal(
            vbt.TRENDLB.run(close_ts, up_th=up_ths, down_th=down_ths, mode="Binary").labels,
            pd.DataFrame(
                np.array(
                    [
                        [1.0, 0.0, 1.0, 0.0, np.nan, np.nan],
                        [0.0, 1.0, 1.0, 0.0, np.nan, np.nan],
                        [1.0, 0.0, 1.0, 0.0, np.nan, np.nan],
                        [1.0, 0.0, 1.0, 0.0, np.nan, np.nan],
                        [np.nan, 1.0, np.nan, 1.0, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("array_0", "array_0", 'binary', "a"),
                        ("array_0", "array_0", 'binary', "b"),
                        ("array_1", "array_1", 'binary', "a"),
                        ("array_1", "array_1", 'binary', "b"),
                        ("array_2", "array_2", 'binary', "a"),
                        ("array_2", "array_2", 'binary', "b"),
                    ],
                    names=["trendlb_up_th", "trendlb_down_th", "trendlb_mode", None],
                ),
            ),
        )
        assert_frame_equal(
            vbt.TRENDLB.run(close_ts, up_th=up_ths, down_th=down_ths, mode="BinaryCont").labels,
            pd.DataFrame(
                np.array(
                    [
                        [1.0, 0.0, 1.0, 0.0, np.nan, np.nan],
                        [0.0, 1.0, 0.5, 0.5, np.nan, np.nan],
                        [1.0, 0.0, 1.0, 0.0, np.nan, np.nan],
                        [0.5, 0.5, 0.5, 0.5, np.nan, np.nan],
                        [np.nan, 1.0, np.nan, 1.0, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("array_0", "array_0", 'binarycont', "a"),
                        ("array_0", "array_0", 'binarycont', "b"),
                        ("array_1", "array_1", 'binarycont', "a"),
                        ("array_1", "array_1", 'binarycont', "b"),
                        ("array_2", "array_2", 'binarycont', "a"),
                        ("array_2", "array_2", 'binarycont', "b"),
                    ],
                    names=["trendlb_up_th", "trendlb_down_th", "trendlb_mode", None],
                ),
            ),
        )
        assert_frame_equal(
            vbt.TRENDLB.run(close_ts, up_th=up_ths, down_th=down_ths, mode="BinaryContSat").labels,
            pd.DataFrame(
                np.array(
                    [
                        [1.0, 0.0, 1.0, 0.0, np.nan, np.nan],
                        [0.0, 1.0, 0.5, 0.4999999999999999, np.nan, np.nan],
                        [1.0, 0.0, 1.0, 0.0, np.nan, np.nan],
                        [0.6666666666666667, 0.0, 0.5, 0.4999999999999999, np.nan, np.nan],
                        [np.nan, 1.0, np.nan, 1.0, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("array_0", "array_0", "binarycontsat", "a"),
                        ("array_0", "array_0", "binarycontsat", "b"),
                        ("array_1", "array_1", "binarycontsat", "a"),
                        ("array_1", "array_1", "binarycontsat", "b"),
                        ("array_2", "array_2", "binarycontsat", "a"),
                        ("array_2", "array_2", "binarycontsat", "b"),
                    ],
                    names=["trendlb_up_th", "trendlb_down_th", "trendlb_mode", None],
                ),
            ),
        )
        assert_frame_equal(
            vbt.TRENDLB.run(close_ts, up_th=up_ths, down_th=down_ths, mode="PctChange").labels,
            pd.DataFrame(
                np.array(
                    [
                        [1.0, -0.3333333333333333, 2.0, -0.6666666666666666, np.nan, np.nan],
                        [-0.5, 0.5, 0.5, -0.5, np.nan, np.nan],
                        [2.0, -0.6666666666666666, 2.0, -0.6666666666666666, np.nan, np.nan],
                        [0.5, -0.5, 0.5, -0.5, np.nan, np.nan],
                        [np.nan, 1.0, np.nan, 1.0, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("array_0", "array_0", "pctchange", "a"),
                        ("array_0", "array_0", "pctchange", "b"),
                        ("array_1", "array_1", "pctchange", "a"),
                        ("array_1", "array_1", "pctchange", "b"),
                        ("array_2", "array_2", "pctchange", "a"),
                        ("array_2", "array_2", "pctchange", "b"),
                    ],
                    names=["trendlb_up_th", "trendlb_down_th", "trendlb_mode", None],
                ),
            ),
        )
        assert_frame_equal(
            vbt.TRENDLB.run(close_ts, up_th=up_ths, down_th=down_ths, mode="PctChangeNorm").labels,
            pd.DataFrame(
                np.array(
                    [
                        [0.5, -0.3333333333333333, 0.6666666666666666, -0.6666666666666666, np.nan, np.nan],
                        [-0.5, 0.3333333333333333, 0.3333333333333333, -0.5, np.nan, np.nan],
                        [
                            0.6666666666666666,
                            -0.6666666666666666,
                            0.6666666666666666,
                            -0.6666666666666666,
                            np.nan,
                            np.nan,
                        ],
                        [0.3333333333333333, -0.5, 0.3333333333333333, -0.5, np.nan, np.nan],
                        [np.nan, 0.5, np.nan, 0.5, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("array_0", "array_0", "pctchangenorm", "a"),
                        ("array_0", "array_0", "pctchangenorm", "b"),
                        ("array_1", "array_1", "pctchangenorm", "a"),
                        ("array_1", "array_1", "pctchangenorm", "b"),
                        ("array_2", "array_2", "pctchangenorm", "a"),
                        ("array_2", "array_2", "pctchangenorm", "b"),
                    ],
                    names=["trendlb_up_th", "trendlb_down_th", "trendlb_mode", None],
                ),
            ),
        )

    def test_BOLB(self):
        assert_frame_equal(
            vbt.BOLB.run(close_ts, window=1, up_th=up_ths, down_th=down_ths).labels,
            pd.DataFrame(
                np.array(
                    [
                        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                        [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
                        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (1, "array_0", "array_0", "a"),
                        (1, "array_0", "array_0", "b"),
                        (1, "array_1", "array_1", "a"),
                        (1, "array_1", "array_1", "b"),
                        (1, "array_2", "array_2", "a"),
                        (1, "array_2", "array_2", "b"),
                    ],
                    names=["bolb_window", "bolb_up_th", "bolb_down_th", None],
                ),
            ),
        )
        assert_frame_equal(
            vbt.BOLB.run(close_ts, window=2, up_th=up_ths, down_th=down_ths).labels,
            pd.DataFrame(
                np.array(
                    [
                        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                        [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
                        [1.0, -1.0, 1.0, -1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (2, "array_0", "array_0", "a"),
                        (2, "array_0", "array_0", "b"),
                        (2, "array_1", "array_1", "a"),
                        (2, "array_1", "array_1", "b"),
                        (2, "array_2", "array_2", "a"),
                        (2, "array_2", "array_2", "b"),
                    ],
                    names=["bolb_window", "bolb_up_th", "bolb_down_th", None],
                ),
            ),
        )
