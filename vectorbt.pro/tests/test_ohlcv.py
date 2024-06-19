import os
from datetime import datetime
import pytest

import vectorbtpro as vbt

from tests.utils import *

ohlcv_ts = pd.DataFrame(
    {
        "open": [1, 2, 3, 4, 5],
        "high": [2.5, 3.5, 4.5, 5.5, 6.5],
        "low": [0.5, 1.5, 2.5, 3.5, 4.5],
        "close": [2, 3, 4, 5, 6],
        "volume": [1, 2, 3, 2, 1],
    },
    index=pd.Index(
        [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 4), datetime(2020, 1, 5)],
    ),
)


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.chunking["n_chunks"] = 2


def teardown_module():
    vbt.settings.reset()


# ############# accessors ############# #


class TestAccessors:
    @pytest.mark.parametrize("test_freq", ["1h", "10h", "3d"])
    def test_resample(self, test_freq):
        assert_frame_equal(
            ohlcv_ts.vbt.ohlcv.resample(test_freq).obj,
            ohlcv_ts.resample(test_freq).agg(
                {
                    "open": lambda x: float(x[0] if len(x) > 0 else np.nan),
                    "high": lambda x: float(x.max() if len(x) > 0 else np.nan),
                    "low": lambda x: float(x.min() if len(x) > 0 else np.nan),
                    "close": lambda x: float(x[-1] if len(x) > 0 else np.nan),
                    "volume": lambda x: float(x.sum() if len(x) > 0 else np.nan),
                }
            ),
        )
