import asyncio
import inspect
import os
from collections import namedtuple
from copy import copy, deepcopy
from datetime import datetime as _datetime, timedelta as _timedelta, time as _time, timezone as _timezone
from functools import wraps
from itertools import product, combinations

import pytest
from numba import njit
from numba.core.registry import CPUDispatcher

import vectorbtpro as vbt
from vectorbtpro.utils import (
    checks,
    config,
    decorators,
    math_,
    array_,
    random_,
    mapping,
    enum_,
    params,
    attr_,
    datetime_,
    schedule_,
    tagging,
    template,
    parsing,
    execution,
    pickling,
    chunking,
    jitting,
)

from tests.utils import *

pathos_available = True
try:
    import pathos
except:
    pathos_available = False

dask_available = True
try:
    import dask
except:
    dask_available = False

ray_available = True
try:
    import ray
except:
    ray_available = False

seed = 42


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    if dask_available:
        dask.config.set(scheduler="synchronous")
    if ray_available:
        ray.init(local_mode=True, num_cpus=1)


def teardown_module():
    if ray_available:
        ray.shutdown()
    vbt.settings.reset()


# ############# config ############# #


class TestConfig:
    def test_copy_dict(self):
        assert config.copy_dict(None) == {}

        def _init_dict():
            return dict(const=0, lst=[1, 2, 3], dct=dict(const=1, lst=[4, 5, 6]))

        dct = _init_dict()
        _dct = config.copy_dict(dct, "shallow", nested=False)
        _dct["const"] = 2
        _dct["dct"]["const"] = 3
        _dct["lst"][0] = 0
        _dct["dct"]["lst"][0] = 0
        assert dct == dict(const=0, lst=[0, 2, 3], dct=dict(const=3, lst=[0, 5, 6]))

        dct = _init_dict()
        _dct = config.copy_dict(dct, "shallow", nested=True)
        _dct["const"] = 2
        _dct["dct"]["const"] = 3
        _dct["lst"][0] = 0
        _dct["dct"]["lst"][0] = 0
        assert dct == dict(const=0, lst=[0, 2, 3], dct=dict(const=1, lst=[0, 5, 6]))

        dct = _init_dict()
        _dct = config.copy_dict(dct, "hybrid", nested=False)
        _dct["const"] = 2
        _dct["dct"]["const"] = 3
        _dct["lst"][0] = 0
        _dct["dct"]["lst"][0] = 0
        assert dct == dict(const=0, lst=[1, 2, 3], dct=dict(const=1, lst=[0, 5, 6]))

        dct = _init_dict()
        _dct = config.copy_dict(dct, "hybrid", nested=True)
        _dct["const"] = 2
        _dct["dct"]["const"] = 3
        _dct["lst"][0] = 0
        _dct["dct"]["lst"][0] = 0
        assert dct == dict(const=0, lst=[1, 2, 3], dct=dict(const=1, lst=[4, 5, 6]))

        def init_config_(**kwargs):
            return config.Config(dict(lst=[1, 2, 3], dct=config.Config(dict(lst=[4, 5, 6]), **kwargs)), **kwargs)

        cfg = init_config_(options_=dict(readonly=True))
        _cfg = config.copy_dict(cfg, "shallow", nested=False)
        assert isinstance(_cfg, config.Config)
        assert _cfg.get_option("readonly")
        assert isinstance(_cfg["dct"], config.Config)
        assert _cfg["dct"].get_option("readonly")
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        assert cfg["lst"] == [0, 2, 3]
        assert cfg["dct"]["lst"] == [0, 5, 6]

        cfg = init_config_(options_=dict(readonly=True))
        _cfg = config.copy_dict(cfg, "shallow", nested=True)
        assert isinstance(_cfg, config.Config)
        assert _cfg.get_option("readonly")
        assert isinstance(_cfg["dct"], config.Config)
        assert _cfg["dct"].get_option("readonly")
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        assert cfg["lst"] == [0, 2, 3]
        assert cfg["dct"]["lst"] == [0, 5, 6]

        cfg = init_config_(options_=dict(readonly=True))
        _cfg = config.copy_dict(cfg, "hybrid", nested=False)
        assert isinstance(_cfg, config.Config)
        assert _cfg.get_option("readonly")
        assert isinstance(_cfg["dct"], config.Config)
        assert _cfg["dct"].get_option("readonly")
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        assert cfg["lst"] == [1, 2, 3]
        assert cfg["dct"]["lst"] == [0, 5, 6]

        cfg = init_config_(options_=dict(readonly=True))
        _cfg = config.copy_dict(cfg, "hybrid", nested=True)
        assert isinstance(_cfg, config.Config)
        assert _cfg.get_option("readonly")
        assert isinstance(_cfg["dct"], config.Config)
        assert _cfg["dct"].get_option("readonly")
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        assert cfg["lst"] == [1, 2, 3]
        assert cfg["dct"]["lst"] == [4, 5, 6]

        cfg = init_config_(options_=dict(readonly=True))
        _cfg = config.copy_dict(cfg, "deep")
        assert isinstance(_cfg, config.Config)
        assert _cfg.get_option("readonly")
        assert isinstance(_cfg["dct"], config.Config)
        assert _cfg["dct"].get_option("readonly")
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        assert cfg["lst"] == [1, 2, 3]
        assert cfg["dct"]["lst"] == [4, 5, 6]

    def test_update_dict(self):
        dct = dict(a=1)
        config.update_dict(dct, None)
        assert dct == dct
        config.update_dict(None, dct)
        assert dct == dct

        def init_config_(**kwargs):
            return config.Config(dict(a=0, b=config.Config(dict(c=1), **kwargs)), **kwargs)

        cfg = init_config_()
        config.update_dict(cfg, dict(a=1), nested=False)
        assert cfg == config.Config(dict(a=1, b=config.Config(dict(c=1))))

        cfg = init_config_()
        config.update_dict(cfg, dict(b=dict(c=2)), nested=False)
        assert cfg == config.Config(dict(a=0, b=dict(c=2)))

        cfg = init_config_()
        config.update_dict(cfg, dict(b=dict(c=2)), nested=True)
        assert cfg == config.Config(dict(a=0, b=config.Config(dict(c=2))))

        cfg = init_config_(options_=dict(readonly=True))
        with pytest.raises(Exception):
            config.update_dict(cfg, dict(b=dict(c=2)), nested=True)

        cfg = init_config_(options_=dict(readonly=True))
        config.update_dict(cfg, dict(b=dict(c=2)), nested=True, force=True)
        assert cfg == config.Config(dict(a=0, b=config.Config(dict(c=2))))
        assert cfg.get_option("readonly")
        assert cfg["b"].get_option("readonly")

        cfg = init_config_(options_=dict(readonly=True))
        config.update_dict(
            cfg,
            config.Config(
                dict(b=config.Config(dict(c=2), options_=dict(readonly=False))),
                options_=dict(readonly=False),
            ),
            nested=True,
            force=True,
        )
        assert cfg == config.Config(dict(a=0, b=config.Config(dict(c=2))))
        assert cfg.get_option("readonly")
        assert cfg["b"].get_option("readonly")

    def test_merge_dicts(self):
        assert config.merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
        assert config.merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}
        assert config.merge_dicts({"a": {"b": 2}}, {"a": {"c": 3}}) == {"a": {"b": 2, "c": 3}}
        assert config.merge_dicts({"a": {"b": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3}}

        def init_configs(**kwargs):
            lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
            return (
                lists,
                config.Config(dict(lst=lists[0], dct=dict(a=1, lst=lists[1])), **kwargs),
                dict(lst=lists[2], dct=config.Config(dict(b=2, lst=lists[3]), **kwargs)),
            )

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=True, copy_mode="shallow", nested=False)
        assert _cfg == dict(lst=lists[2], dct=config.Config(dict(b=2, lst=lists[3])))
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [0, 8, 9]
        assert _cfg["dct"]["lst"] == [0, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=True, copy_mode="shallow", nested=True)
        assert _cfg == dict(lst=lists[2], dct=dict(a=1, b=2, lst=lists[3]))
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [0, 8, 9]
        assert _cfg["dct"]["lst"] == [0, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        cfg2["dct"] = config.atomic_dict(cfg2["dct"])
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=True, copy_mode="shallow", nested=True)
        assert _cfg == dict(lst=lists[2], dct=config.atomic_dict(b=2, lst=lists[3]))
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [0, 8, 9]
        assert _cfg["dct"]["lst"] == [0, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, config.atomic_dict(cfg2), to_dict=True, copy_mode="shallow", nested=True)
        assert _cfg == config.atomic_dict(lst=lists[2], dct=dict(b=2, lst=lists[3]))
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [0, 8, 9]
        assert _cfg["dct"]["lst"] == [0, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=False, copy_mode="shallow", nested=False)
        assert _cfg == config.Config(dict(lst=lists[2], dct=config.Config(dict(b=2, lst=lists[3]))))
        assert _cfg.get_option("readonly")
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [0, 8, 9]
        assert _cfg["dct"]["lst"] == [0, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=False, copy_mode="hybrid", nested=False)
        assert _cfg == config.Config(dict(lst=lists[2], dct=config.Config(dict(b=2, lst=lists[3]))))
        assert _cfg.get_option("readonly")
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [7, 8, 9]
        assert _cfg["dct"]["lst"] == [0, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=False, copy_mode="hybrid", nested=True)
        assert _cfg == config.Config(dict(lst=lists[2], dct=dict(a=1, b=2, lst=lists[3])))
        assert _cfg.get_option("readonly")
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [7, 8, 9]
        assert _cfg["dct"]["lst"] == [10, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=False, copy_mode="deep", nested=False)
        assert _cfg == config.Config(dict(lst=lists[2], dct=config.Config(dict(b=2, lst=lists[3]))))
        assert _cfg.get_option("readonly")
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [7, 8, 9]
        assert _cfg["dct"]["lst"] == [10, 11, 12]

    def test_config_copy(self):
        def init_config(**kwargs):
            dct = dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6])))
            return dct, config.Config(dct, **kwargs)

        dct, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="shallow"), nested=False))
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        dct["lst"][0] = 0
        dct["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[0, 2, 3], dct=config.Config(dict(const=3, lst=[0, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[0, 2, 3], dct=config.Config(dict(const=3, lst=[0, 5, 6]))
        )

        dct, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="shallow"), nested=True))
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        dct["lst"][0] = 0
        dct["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[0, 2, 3], dct=config.Config(dict(const=1, lst=[0, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[0, 2, 3], dct=config.Config(dict(const=1, lst=[0, 5, 6]))
        )

        dct, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="hybrid"), nested=True))
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        dct["lst"][0] = 0
        dct["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

        dct, cfg = init_config(
            options_=dict(
                copy_kwargs=dict(copy_mode="shallow"),
                reset_dct_copy_kwargs=dict(copy_mode="hybrid"),
                nested=True,
            )
        )
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        dct["lst"][0] = 0
        dct["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[0, 2, 3], dct=config.Config(dict(const=1, lst=[0, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

        dct, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="deep"), nested=True))
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        dct["lst"][0] = 0
        dct["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

        init_d, _ = init_config()
        init_d = config.copy_dict(init_d, "deep")
        dct, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="hybrid"), reset_dct=init_d, nested=True))
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        init_d["lst"][0] = 0
        init_d["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

        init_d, _ = init_config()
        init_d = config.copy_dict(init_d, "deep")
        dct, cfg = init_config(
            options_=dict(
                copy_kwargs=dict(copy_mode="hybrid"),
                reset_dct=init_d,
                reset_dct_copy_kwargs=dict(copy_mode="shallow"),
                nested=True,
            )
        )
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        dct["lst"][0] = 0
        dct["dct"]["lst"][0] = 0
        init_d["const"] = 2
        init_d["dct"]["const"] = 3
        init_d["lst"][0] = 0
        init_d["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[0, 2, 3], dct=config.Config(dict(const=1, lst=[0, 5, 6]))
        )

        _, cfg = init_config(options_=dict(nested=True))
        _cfg = copy(cfg)
        _cfg["const"] = 2
        _cfg["dct"]["const"] = 3
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        _cfg.get_option("reset_dct")["const"] = 2
        _cfg.get_option("reset_dct")["dct"]["const"] = 3
        _cfg.get_option("reset_dct")["lst"][0] = 0
        _cfg.get_option("reset_dct")["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[0, 2, 3], dct=config.Config(dict(const=3, lst=[0, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=2, lst=[0, 2, 3], dct=config.Config(dict(const=3, lst=[0, 5, 6]))
        )

        _, cfg = init_config(options_=dict(nested=True))
        _cfg = deepcopy(cfg)
        _cfg["const"] = 2
        _cfg["dct"]["const"] = 3
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        _cfg.get_option("reset_dct")["const"] = 2
        _cfg.get_option("reset_dct")["dct"]["const"] = 3
        _cfg.get_option("reset_dct")["lst"][0] = 0
        _cfg.get_option("reset_dct")["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

        _, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="hybrid"), nested=True))
        _cfg = cfg.copy()
        _cfg["const"] = 2
        _cfg["dct"]["const"] = 3
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        _cfg.get_option("reset_dct")["const"] = 2
        _cfg.get_option("reset_dct")["dct"]["const"] = 3
        _cfg.get_option("reset_dct")["lst"][0] = 0
        _cfg.get_option("reset_dct")["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

        _, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="hybrid"), nested=True))
        _cfg = cfg.copy(reset_dct_copy_kwargs=dict(copy_mode="shallow"))
        _cfg["const"] = 2
        _cfg["dct"]["const"] = 3
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        _cfg.get_option("reset_dct")["const"] = 2
        _cfg.get_option("reset_dct")["dct"]["const"] = 3
        _cfg.get_option("reset_dct")["lst"][0] = 0
        _cfg.get_option("reset_dct")["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[0, 2, 3], dct=config.Config(dict(const=1, lst=[0, 5, 6]))
        )

        _, cfg = init_config(options_=dict(nested=True))
        _cfg = cfg.copy(copy_mode="deep")
        _cfg["const"] = 2
        _cfg["dct"]["const"] = 3
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        _cfg.get_option("reset_dct")["const"] = 2
        _cfg.get_option("reset_dct")["dct"]["const"] = 3
        _cfg.get_option("reset_dct")["lst"][0] = 0
        _cfg.get_option("reset_dct")["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

    def test_config_convert_children(self):
        cfg = config.Config(
            dict(dct=config.child_dict(dct=config.Config(dict(), options_=dict(nested=False)))),
            options_=dict(nested=True, convert_children=True),
        )
        assert cfg.get_option("nested")
        assert cfg.get_option("convert_children")
        assert isinstance(cfg["dct"], config.Config)
        assert cfg["dct"].get_option("nested")
        assert cfg["dct"].get_option("convert_children")
        assert isinstance(cfg["dct"]["dct"], config.Config)
        assert not cfg["dct"]["dct"].get_option("nested")
        assert not cfg["dct"]["dct"].get_option("convert_children")

    def test_config_from_config(self):
        cfg = config.Config(
            config.Config(
                dict(a=0),
                options_=dict(
                    copy_kwargs=dict(copy_mode="deep", nested=True),
                    reset_dct=dict(b=0),
                    reset_dct_copy_kwargs=dict(copy_mode="deep", nested=True),
                    frozen_keys=True,
                    readonly=True,
                    nested=True,
                    convert_children=True,
                    as_attrs=True,
                ),
            )
        )
        assert dict(cfg) == dict(a=0)
        assert cfg.get_option("copy_kwargs") == dict(copy_mode="deep", nested=True)
        assert cfg.get_option("reset_dct") == dict(b=0)
        assert cfg.get_option("reset_dct_copy_kwargs") == dict(copy_mode="deep", nested=True)
        assert cfg.get_option("frozen_keys")
        assert cfg.get_option("readonly")
        assert cfg.get_option("nested")
        assert cfg.get_option("convert_children")
        assert cfg.get_option("as_attrs")

        c2 = config.Config(
            cfg,
            options_=dict(
                copy_kwargs=dict(copy_mode="hybrid"),
                reset_dct=dict(b=0),
                reset_dct_copy_kwargs=dict(nested=False),
                frozen_keys=False,
                readonly=False,
                nested=False,
                convert_children=False,
                as_attrs=False,
            ),
        )
        assert dict(c2) == dict(a=0)
        assert c2.get_option("copy_kwargs") == dict(copy_mode="hybrid", nested=True)
        assert c2.get_option("reset_dct") == dict(b=0)
        assert c2.get_option("reset_dct_copy_kwargs") == dict(copy_mode="hybrid", nested=False)
        assert not c2.get_option("frozen_keys")
        assert not c2.get_option("readonly")
        assert not c2.get_option("nested")
        assert not c2.get_option("convert_children")
        assert not c2.get_option("as_attrs")

    def test_config_defaults(self):
        cfg = config.Config(dict(a=0))
        assert dict(cfg) == dict(a=0)
        assert cfg.get_option("copy_kwargs") == dict(copy_mode="none", nested=True)
        assert cfg.get_option("reset_dct") == dict(a=0)
        assert cfg.get_option("reset_dct_copy_kwargs") == dict(copy_mode="hybrid", nested=True)
        assert not cfg.get_option("frozen_keys")
        assert not cfg.get_option("readonly")
        assert cfg.get_option("nested")
        assert not cfg.get_option("convert_children")
        assert not cfg.get_option("as_attrs")

        vbt.settings.config.options.reset()
        vbt.settings.config.options["copy_kwargs"] = dict(copy_mode="deep")
        vbt.settings.config.options["reset_dct_copy_kwargs"] = dict(copy_mode="deep")
        vbt.settings.config.options["frozen_keys"] = True
        vbt.settings.config.options["readonly"] = True
        vbt.settings.config.options["nested"] = False
        vbt.settings.config.options["convert_children"] = True
        vbt.settings.config.options["as_attrs"] = True

        cfg = config.Config(dict(a=0))
        assert dict(cfg) == dict(a=0)
        assert cfg.get_option("copy_kwargs") == dict(copy_mode="deep", nested=False)
        assert cfg.get_option("reset_dct") == dict(a=0)
        assert cfg.get_option("reset_dct_copy_kwargs") == dict(copy_mode="deep", nested=False)
        assert cfg.get_option("frozen_keys")
        assert cfg.get_option("readonly")
        assert not cfg.get_option("nested")
        assert cfg.get_option("convert_children")
        assert cfg.get_option("as_attrs")

        vbt.settings.config.reset()

    def test_config_as_attrs(self):
        cfg = config.Config(dict(a=0, b=0, dct=dict(d=0)), options_=dict(as_attrs=True))
        assert cfg.a == 0
        assert cfg.b == 0
        with pytest.raises(Exception):
            assert cfg.dct.d == 0

        cfg.e = 0
        assert cfg["e"] == 0
        cfg["f"] = 0
        assert cfg.f == 0
        with pytest.raises(Exception):
            assert cfg.g == 0
        del cfg["f"]
        with pytest.raises(Exception):
            assert cfg.f == 0
        del cfg.e
        with pytest.raises(Exception):
            assert cfg["e"] == 0
        cfg.clear()
        assert dict(cfg) == dict()
        assert not hasattr(cfg, "a")
        assert not hasattr(cfg, "b")
        cfg.a = 0
        cfg.b = 0
        cfg.pop("a")
        assert not hasattr(cfg, "a")
        cfg.popitem()
        assert not hasattr(cfg, "b")

        cfg = config.Config(
            config.child_dict(a=0, b=0, dct=config.child_dict(d=0)),
            options_=dict(as_attrs=True, nested=True, convert_children=True),
        )
        assert cfg.a == 0
        assert cfg.b == 0
        assert cfg.dct.d == 0

        with pytest.raises(Exception):
            config.Config(dict(options_=True), options_=dict(as_attrs=True))

    def test_config_frozen_keys(self):
        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg.pop("a")
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg.popitem()
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg.clear()
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg.update(dict(a=1))
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg.update(dict(b=0))
        assert dict(cfg) == dict(a=0, b=0)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        del cfg["a"]
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg["a"] = 1
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg["b"] = 0
        assert dict(cfg) == dict(a=0, b=0)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        with pytest.raises(Exception):
            cfg.pop("a")
        cfg.pop("a", force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        with pytest.raises(Exception):
            cfg.popitem()
        cfg.popitem(force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        with pytest.raises(Exception):
            cfg.clear()
        cfg.clear(force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        cfg.update(dict(a=1))
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        with pytest.raises(Exception):
            cfg.update(dict(b=0))
        cfg.update(dict(b=0), force=True)
        assert dict(cfg) == dict(a=0, b=0)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        with pytest.raises(Exception):
            del cfg["a"]
        cfg.__delitem__("a", force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        cfg["a"] = 1
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        with pytest.raises(Exception):
            cfg["b"] = 0
        cfg.__setitem__("b", 0, force=True)
        assert dict(cfg) == dict(a=0, b=0)

    def test_config_readonly(self):
        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg.pop("a")
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg.popitem()
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg.clear()
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg.update(dict(a=1))
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg.update(dict(b=0))
        assert dict(cfg) == dict(a=0, b=0)

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        del cfg["a"]
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg["a"] = 1
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg["b"] = 0
        assert dict(cfg) == dict(a=0, b=0)

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg.pop("a")
        cfg.pop("a", force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg.popitem()
        cfg.popitem(force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg.clear()
        cfg.clear(force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg.update(dict(a=1))
        cfg.update(dict(a=1), force=True)
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg.update(dict(b=0))
        cfg.update(dict(b=0), force=True)
        assert dict(cfg) == dict(a=0, b=0)

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            del cfg["a"]
        cfg.__delitem__("a", force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg["a"] = 1
        cfg.__setitem__("a", 1, force=True)
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg["b"] = 0
        cfg.__setitem__("b", 0, force=True)
        assert dict(cfg) == dict(a=0, b=0)

    def test_config_merge_with(self):
        cfg1 = config.Config(
            dict(a=0, dct=dict(b=1, dct=config.Config(dict(c=2), options_=dict(readonly=False)))),
            options_=dict(readonly=False, nested=False),
        )
        cfg2 = config.Config(
            dict(d=3, dct=config.Config(dict(e=4, dct=dict(f=5)), options_=dict(readonly=True))),
            options_=dict(readonly=True, nested=False),
        )
        _cfg = cfg1.merge_with(cfg2)
        assert _cfg == dict(a=0, d=3, dct=cfg2["dct"])
        assert not isinstance(_cfg, config.Config)
        assert isinstance(_cfg["dct"], config.Config)
        assert not isinstance(_cfg["dct"]["dct"], config.Config)

        _cfg = cfg1.merge_with(cfg2, to_dict=False, nested=False)
        assert _cfg == config.Config(dict(a=0, d=3, dct=cfg2["dct"]))
        assert not _cfg.get_option("readonly")
        assert isinstance(_cfg["dct"], config.Config)
        assert _cfg["dct"].get_option("readonly")
        assert not isinstance(_cfg["dct"]["dct"], config.Config)

        _cfg = cfg1.merge_with(cfg2, to_dict=False, nested=True)
        assert _cfg == config.Config(dict(a=0, d=3, dct=dict(b=1, e=4, dct=config.Config(dict(c=2, f=5)))))
        assert not _cfg.get_option("readonly")
        assert not isinstance(_cfg["dct"], config.Config)
        assert isinstance(_cfg["dct"]["dct"], config.Config)
        assert not _cfg["dct"]["dct"].get_option("readonly")

    def test_config_reset(self):
        cfg = config.Config(
            dict(a=0, dct=dict(b=0)),
            options_=dict(copy_kwargs=dict(copy_mode="shallow"), nested=False),
        )
        cfg["a"] = 1
        cfg["dct"]["b"] = 1
        cfg.reset()
        assert cfg == config.Config(dict(a=0, dct=dict(b=1)))

        cfg = config.Config(
            dict(a=0, dct=dict(b=0)),
            options_=dict(copy_kwargs=dict(copy_mode="hybrid"), nested=False),
        )
        cfg["a"] = 1
        cfg["dct"]["b"] = 1
        cfg.reset()
        assert cfg == config.Config(dict(a=0, dct=dict(b=0)))

        cfg = config.Config(
            dict(a=0, dct=dict(b=0)),
            options_=dict(copy_kwargs=dict(copy_mode="deep"), nested=False),
        )
        cfg["a"] = 1
        cfg["dct"]["b"] = 1
        cfg.reset()
        assert cfg == config.Config(dict(a=0, dct=dict(b=0)))

        cfg = config.Config(
            dict(a=0, dct=dict(b=0)),
            options_=dict(copy_kwargs=dict(copy_mode="shallow"), nested=True),
        )
        cfg["a"] = 1
        cfg["dct"]["b"] = 1
        cfg.reset()
        assert cfg == config.Config(dict(a=0, dct=dict(b=0)))

        cfg = config.Config(
            dict(a=0, dct=dict(b=0)),
            options_=dict(copy_kwargs=dict(copy_mode="hybrid"), nested=True),
        )
        cfg["a"] = 1
        cfg["dct"]["b"] = 1
        cfg.reset()
        assert cfg == config.Config(dict(a=0, dct=dict(b=0)))

        cfg = config.Config(
            dict(a=0, dct=dict(b=0)),
            options_=dict(copy_kwargs=dict(copy_mode="deep"), nested=True),
        )
        cfg["a"] = 1
        cfg["dct"]["b"] = 1
        cfg.reset()
        assert cfg == config.Config(dict(a=0, dct=dict(b=0)))

    def test_config_save_and_load(self, tmp_path):
        cfg = config.Config(
            dict(a=0, dct=dict(b=[1, 2, 3], dct=config.Config(options_=dict(readonly=False)))),
            options_=dict(
                copy_kwargs=dict(copy_mode="deep", nested=True),
                reset_dct=dict(b=0),
                reset_dct_copy_kwargs=dict(copy_mode="deep", nested=True),
                pickle_reset_dct=True,
                frozen_keys=True,
                readonly=True,
                nested=True,
                convert_children=True,
                as_attrs=True,
            ),
        )
        cfg.save(tmp_path / "config")
        new_cfg = config.Config.load(tmp_path / "config")
        assert new_cfg == deepcopy(cfg)
        assert new_cfg.__dict__ == deepcopy(cfg).__dict__
        cfg.save(tmp_path / "config", file_format="ini")
        new_cfg = config.Config.load(tmp_path / "config", file_format="ini")
        assert new_cfg == deepcopy(cfg)
        assert new_cfg.__dict__ == deepcopy(cfg).__dict__

    def test_config_load_update(self, tmp_path):
        cfg1 = config.Config(
            dict(a=0, dct=dict(b=[1, 2, 3], dct=config.Config(options_=dict(readonly=False)))),
            options_=dict(
                copy_kwargs=dict(copy_mode="deep", nested=True),
                reset_dct=dict(b=0),
                reset_dct_copy_kwargs=dict(copy_mode="deep", nested=True),
                pickle_reset_dct=True,
                frozen_keys=True,
                readonly=True,
                nested=True,
                convert_children=True,
                as_attrs=True,
            ),
        )
        cfg2 = cfg3 = cfg4 = cfg5 = config.Config(
            dct=dict(a=1, dct=dict(b=[4, 5, 6], dct=config.Config(options_=dict(readonly=True)))),
            options_=dict(
                copy_kwargs=dict(copy_mode="shallow", nested=False),
                reset_dct=dict(b=1),
                reset_dct_copy_kwargs=dict(copy_mode="shallow", nested=False),
                pickle_reset_dct=False,
                frozen_keys=False,
                readonly=False,
                nested=False,
                convert_children=False,
                as_attrs=False,
            ),
        )
        cfg2 = deepcopy(cfg2)
        cfg3 = deepcopy(cfg3)
        cfg4 = deepcopy(cfg4)
        cfg5 = deepcopy(cfg5)
        cfg1.save(tmp_path / "config")
        cfg2.load_update(tmp_path / "config")
        assert cfg2 == deepcopy(cfg1)
        assert cfg2.__dict__ != cfg1.__dict__
        cfg3.load_update(tmp_path / "config", update_options=True)
        assert cfg3 == deepcopy(cfg1)
        assert cfg3.__dict__ == cfg1.__dict__
        cfg1.save(tmp_path / "config", file_format="ini")
        cfg4.load_update(tmp_path / "config", file_format="ini")
        assert cfg4 == deepcopy(cfg1)
        assert cfg4.__dict__ != cfg1.__dict__
        cfg5.load_update(tmp_path / "config", file_format="ini", update_options=True)
        assert cfg5 == deepcopy(cfg1)
        assert cfg5.__dict__ == cfg1.__dict__

    def test_configured(self, tmp_path):
        class H(config.Configured):
            _rec_id = "123456789"
            _writeable_attrs = {"my_attr", "my_cfg"}

            def __init__(self, a, b=2, **kwargs):
                super().__init__(a=a, b=b, **kwargs)
                self.my_attr = 100
                self.my_cfg = config.Config(dict(sr=pd.Series([1, 2, 3])))

        assert H(1).config == config.Config({"a": 1, "b": 2})
        assert H(1).replace(b=3).config == config.Config({"a": 1, "b": 3})
        assert H(pd.Series([1, 2, 3])) == H(pd.Series([1, 2, 3]))
        assert H(pd.Series([1, 2, 3])) != H(pd.Series([1, 2, 4]))
        assert H(pd.DataFrame([1, 2, 3])) == H(pd.DataFrame([1, 2, 3]))
        assert H(pd.DataFrame([1, 2, 3])) != H(pd.DataFrame([1, 2, 4]))
        assert H(pd.Index([1, 2, 3])) == H(pd.Index([1, 2, 3]))
        assert H(pd.Index([1, 2, 3])) != H(pd.Index([1, 2, 4]))
        assert H(np.array([1, 2, 3])) == H(np.array([1, 2, 3]))
        assert H(np.array([1, 2, 3])) != H(np.array([1, 2, 4]))
        assert H(None) == H(None)
        assert H(None) != H(10.0)

        vbt.RecInfo(H._rec_id, H).register()

        h = H(1)
        h.my_attr = 200
        h.my_cfg["df"] = pd.DataFrame([1, 2, 3])
        h2 = H(1)
        h2.my_attr = 200
        h2.my_cfg["df"] = pd.DataFrame([1, 2, 3])
        h.save(tmp_path / "configured")
        new_h = H.load(tmp_path / "configured")
        assert new_h == h2
        assert new_h != H(1)
        assert new_h.__dict__ == h2.__dict__
        assert new_h.__dict__ != H(1).__dict__
        assert new_h.my_attr == h.my_attr
        assert new_h.my_cfg == h.my_cfg
        h.save(tmp_path / "configured", file_format="ini")
        new_h = H.load(tmp_path / "configured", file_format="ini")
        assert new_h == h2
        assert new_h != H(1)
        assert new_h.__dict__ == h2.__dict__
        assert new_h.__dict__ != H(1).__dict__
        assert new_h.my_attr == h.my_attr
        assert new_h.my_cfg == h.my_cfg


# ############# decorators ############# #


class TestDecorators:
    def test_class_or_instancemethod(self):
        class G:
            @decorators.class_or_instancemethod
            def g(cls_or_self):
                if isinstance(cls_or_self, type):
                    return True  # class
                return False  # instance

        assert G.g()
        assert not G().g()

    def test_class_or_instanceproperty(self):
        class G:
            @decorators.class_or_instanceproperty
            def g(cls_or_self):
                if isinstance(cls_or_self, type):
                    return True  # class
                return False  # instance

        assert G.g
        assert not G().g

    def test_custom_property(self):
        class G:
            @decorators.custom_property(some="key")
            def cache_me(self):
                return np.random.uniform()

        assert "some" in G.cache_me.options
        assert G.cache_me.options["some"] == "key"

    def test_custom_function(self):
        @decorators.custom_function(some="key")
        def cache_me():
            return np.random.uniform()

        assert "some" in cache_me.options
        assert cache_me.options["some"] == "key"


# ############# attr_ ############# #


class TestAttr:
    def test_deep_getattr(self):
        class A:
            def a(self, x, y=None):
                return x + y

        class B:
            def a(self):
                return A()

            def b(self, x):
                return x

            @property
            def b_prop(self):
                return 1

        class C:
            @property
            def b(self):
                return B()

            @property
            def c(self):
                return 0

        with pytest.raises(Exception):
            attr_.deep_getattr(A(), "a")
        with pytest.raises(Exception):
            attr_.deep_getattr(A(), ("a",))
        with pytest.raises(Exception):
            attr_.deep_getattr(A(), ("a", 1))
        with pytest.raises(Exception):
            attr_.deep_getattr(A(), ("a", (1,)))
        assert attr_.deep_getattr(A(), ("a", (1,), {"y": 1})) == 2
        assert attr_.deep_getattr(C(), "c") == 0
        assert attr_.deep_getattr(C(), ["c"]) == 0
        assert attr_.deep_getattr(C(), ["b", ("b", (1,))]) == 1
        assert attr_.deep_getattr(C(), "b.b(1)") == 1
        assert attr_.deep_getattr(C(), ["b", ("a",), ("a", (1,), {"y": 1})]) == 2
        assert attr_.deep_getattr(C(), "b.a().a(1, y=1)") == 2
        assert attr_.deep_getattr(C(), "b.b_prop") == 1
        assert callable(attr_.deep_getattr(C(), "b.a.a", call_last_attr=False))


# ############# checks ############# #


class TestChecks:
    def test_is_np_array(self):
        assert not checks.is_np_array(0)
        assert checks.is_np_array(np.array([0]))
        assert not checks.is_np_array(pd.Series([1, 2, 3]))
        assert not checks.is_np_array(pd.DataFrame([1, 2, 3]))

    def test_is_pandas(self):
        assert not checks.is_pandas(0)
        assert not checks.is_pandas(np.array([0]))
        assert checks.is_pandas(pd.Series([1, 2, 3]))
        assert checks.is_pandas(pd.DataFrame([1, 2, 3]))

    def test_is_series(self):
        assert not checks.is_series(0)
        assert not checks.is_series(np.array([0]))
        assert checks.is_series(pd.Series([1, 2, 3]))
        assert not checks.is_series(pd.DataFrame([1, 2, 3]))

    def test_is_frame(self):
        assert not checks.is_frame(0)
        assert not checks.is_frame(np.array([0]))
        assert not checks.is_frame(pd.Series([1, 2, 3]))
        assert checks.is_frame(pd.DataFrame([1, 2, 3]))

    def test_is_array(self):
        assert not checks.is_any_array(0)
        assert checks.is_any_array(np.array([0]))
        assert checks.is_any_array(pd.Series([1, 2, 3]))
        assert checks.is_any_array(pd.DataFrame([1, 2, 3]))

    def test_is_sequence(self):
        assert checks.is_sequence([1, 2, 3])
        assert checks.is_sequence("123")
        assert not checks.is_sequence(0)
        assert not checks.is_sequence(dict(a=2).items())

    def test_is_iterable(self):
        assert checks.is_iterable([1, 2, 3])
        assert checks.is_iterable("123")
        assert not checks.is_iterable(0)
        assert checks.is_iterable(dict(a=2).items())

    def test_is_numba_func(self):
        def test_func(x):
            return x

        @njit
        def test_func_nb(x):
            return x

        assert not checks.is_numba_func(test_func)
        assert checks.is_numba_func(test_func_nb)

    def test_is_hashable(self):
        assert checks.is_hashable(2)
        assert not checks.is_hashable(np.asarray(2))

    def test_is_index_equal(self):
        assert checks.is_index_equal(pd.Index([0]), pd.Index([0]))
        assert not checks.is_index_equal(pd.Index([0]), pd.Index([1]))
        assert not checks.is_index_equal(pd.Index([0], name="name"), pd.Index([0]))
        assert checks.is_index_equal(pd.Index([0], name="name"), pd.Index([0]), check_names=False)
        assert not checks.is_index_equal(pd.MultiIndex.from_arrays([[0], [1]]), pd.Index([0]))
        assert checks.is_index_equal(pd.MultiIndex.from_arrays([[0], [1]]), pd.MultiIndex.from_arrays([[0], [1]]))
        assert checks.is_index_equal(
            pd.MultiIndex.from_arrays([[0], [1]], names=["name1", "name2"]),
            pd.MultiIndex.from_arrays([[0], [1]], names=["name1", "name2"]),
        )
        assert not checks.is_index_equal(
            pd.MultiIndex.from_arrays([[0], [1]], names=["name1", "name2"]),
            pd.MultiIndex.from_arrays([[0], [1]], names=["name3", "name4"]),
        )

    def test_is_default_index(self):
        assert checks.is_default_index(pd.DataFrame([[1, 2, 3]]).columns)
        assert checks.is_default_index(pd.Series([1, 2, 3]).to_frame().columns)
        assert checks.is_default_index(pd.Index([0, 1, 2]))
        assert not checks.is_default_index(pd.Index([0, 1, 2], name="name"))

    def test_is_equal(self):
        assert checks.is_equal(np.arange(3), np.arange(3), np.array_equal)
        assert not checks.is_equal(np.arange(3), None, np.array_equal)
        assert not checks.is_equal(None, np.arange(3), np.array_equal)
        assert checks.is_equal(None, None, np.array_equal)

    def test_is_namedtuple(self):
        assert checks.is_namedtuple(namedtuple("Hello", ["world"])(*range(1)))
        assert not checks.is_namedtuple((0,))

    def test_func_accepts_arg(self):
        def test(a, *args, b=2, **kwargs):
            pass

        assert checks.func_accepts_arg(test, "a")
        assert not checks.func_accepts_arg(test, "args")
        assert checks.func_accepts_arg(test, "*args")
        assert checks.func_accepts_arg(test, "b")
        assert not checks.func_accepts_arg(test, "kwargs")
        assert checks.func_accepts_arg(test, "**kwargs")
        assert not checks.func_accepts_arg(test, "c")

    def test_is_deep_equal(self):
        sr = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="index"), name="name")
        sr2 = pd.Series([1.0, 2.0, 3.0], index=sr.index, name=sr.name)
        sr3 = pd.Series([np.nan, 2.0, 3.0], index=sr.index, name=sr.name)
        sr4 = pd.Series([np.nan, 2.0, 3.0 + 1e-15], index=sr.index, name=sr.name)
        assert checks.is_deep_equal(sr, sr.copy())
        assert checks.is_deep_equal(sr2, sr2.copy())
        assert checks.is_deep_equal(sr3, sr3.copy())
        assert checks.is_deep_equal(sr4, sr4.copy())
        assert not checks.is_deep_equal(sr, sr2)
        assert checks.is_deep_equal(sr3, sr4)
        assert not checks.is_deep_equal(sr3, sr4, rtol=0, atol=1e-16)
        assert not checks.is_deep_equal(sr3, sr4, check_exact=True)
        assert not checks.is_deep_equal(sr, sr.rename("name2"))
        assert checks.is_deep_equal(sr.index, sr.copy().index)
        assert not checks.is_deep_equal(sr.index, sr.copy().index[:-1])
        assert not checks.is_deep_equal(sr.index, sr.copy().rename("indx2"))
        assert checks.is_deep_equal(sr.to_frame(), sr.to_frame().copy())
        assert not checks.is_deep_equal(sr, sr.to_frame().copy())
        assert not checks.is_deep_equal(sr.to_frame(), sr.copy())

        arr = np.array([1, 2, 3])
        arr2 = np.array([1.0, 2.0, 3.0])
        arr3 = np.array([np.nan, 2.0, 3.0])
        arr4 = np.array([np.nan, 2.0, 3 + 1e-15])
        assert checks.is_deep_equal(arr, arr.copy())
        assert checks.is_deep_equal(arr2, arr2.copy())
        assert checks.is_deep_equal(arr3, arr3.copy())
        assert checks.is_deep_equal(arr4, arr4.copy())
        assert not checks.is_deep_equal(arr, arr2)
        assert checks.is_deep_equal(arr3, arr4)
        assert not checks.is_deep_equal(arr3, arr4, rtol=0, atol=1e-16)
        assert not checks.is_deep_equal(arr3, arr4, check_exact=True)

        records_arr = np.asarray(
            [
                (1, 1.0),
                (2, 2.0),
                (3, 3.0),
            ],
            dtype=np.dtype([("a", np.int32), ("b", np.float64)]),
        )
        records_arr2 = np.asarray(
            [
                (1.0, 1.0),
                (2.0, 2.0),
                (3.0, 3.0),
            ],
            dtype=np.dtype([("a", np.float64), ("b", np.float64)]),
        )
        records_arr3 = np.asarray(
            [
                (np.nan, 1.0),
                (2.0, 2.0),
                (3.0, 3.0),
            ],
            dtype=np.dtype([("a", np.float64), ("b", np.float64)]),
        )
        records_arr4 = np.asarray(
            [
                (np.nan, 1.0),
                (2.0, 2.0),
                (3.0 + 1e-15, 3.0),
            ],
            dtype=np.dtype([("a", np.float64), ("b", np.float64)]),
        )
        assert checks.is_deep_equal(records_arr, records_arr.copy())
        assert checks.is_deep_equal(records_arr2, records_arr2.copy())
        assert checks.is_deep_equal(records_arr3, records_arr3.copy())
        assert checks.is_deep_equal(records_arr4, records_arr4.copy())
        assert not checks.is_deep_equal(records_arr, records_arr2)
        assert checks.is_deep_equal(records_arr3, records_arr4)
        assert not checks.is_deep_equal(records_arr3, records_arr4, rtol=0, atol=1e-16)
        assert not checks.is_deep_equal(records_arr3, records_arr4, check_exact=True)

        assert checks.is_deep_equal([sr, arr, records_arr], [sr, arr, records_arr])
        assert not checks.is_deep_equal([sr, arr, records_arr], [sr, arr, records_arr2])
        assert not checks.is_deep_equal([sr, arr, records_arr], [sr, records_arr, arr])
        assert checks.is_deep_equal(
            {"sr": sr, "arr": arr, "records_arr": records_arr},
            {"sr": sr, "arr": arr, "records_arr": records_arr},
        )
        assert not checks.is_deep_equal(
            {"sr": sr, "arr": arr, "records_arr": records_arr},
            {"sr": sr, "arr": arr, "records_arr2": records_arr},
        )
        assert not checks.is_deep_equal(
            {"sr": sr, "arr": arr, "records_arr": records_arr},
            {"sr": sr, "arr": arr, "records_arr": records_arr2},
        )

        assert checks.is_deep_equal(0, 0)
        assert not checks.is_deep_equal(0, False)
        assert not checks.is_deep_equal(0, 1)
        assert checks.is_deep_equal(lambda x: x, lambda x: x)
        assert not checks.is_deep_equal(lambda x: x, lambda x: 2 * x)

    def test_is_instance_of(self):
        class _A:
            pass

        class A:
            pass

        class B:
            pass

        class C(B):
            pass

        class D(A, C):
            pass

        d = D()

        assert not checks.is_instance_of(d, _A)
        assert checks.is_instance_of(d, A)
        assert checks.is_instance_of(d, B)
        assert checks.is_instance_of(d, C)
        assert checks.is_instance_of(d, D)
        assert checks.is_instance_of(d, object)

        assert not checks.is_instance_of(d, "_A")
        assert checks.is_instance_of(d, "A")
        assert checks.is_instance_of(d, "B")
        assert checks.is_instance_of(d, "C")
        assert checks.is_instance_of(d, "D")
        assert checks.is_instance_of(d, "object")

    def test_is_subclass_of(self):
        class _A:
            pass

        class A:
            pass

        class B:
            pass

        class C(B):
            pass

        class D(A, C):
            pass

        assert not checks.is_subclass_of(D, _A)
        assert checks.is_subclass_of(D, A)
        assert checks.is_subclass_of(D, B)
        assert checks.is_subclass_of(D, C)
        assert checks.is_subclass_of(D, D)
        assert checks.is_subclass_of(D, object)

        assert not checks.is_subclass_of(D, "_A")
        assert checks.is_subclass_of(D, "A")
        assert checks.is_subclass_of(D, "B")
        assert checks.is_subclass_of(D, "C")
        assert checks.is_subclass_of(D, "D")
        assert checks.is_subclass_of(D, "object")

        assert not checks.is_subclass_of(D, vbt.Regex("_A"))
        assert checks.is_subclass_of(D, vbt.Regex("[A-D]"))
        assert not checks.is_subclass_of(D, vbt.Regex("[E-F]"))
        assert checks.is_subclass_of(D, vbt.Regex("object"))

    def test_assert_in(self):
        checks.assert_in(0, (0, 1))
        with pytest.raises(Exception):
            checks.assert_in(2, (0, 1))

    def test_assert_numba_func(self):
        def test_func(x):
            return x

        @njit
        def test_func_nb(x):
            return x

        checks.assert_numba_func(test_func_nb)
        with pytest.raises(Exception):
            checks.assert_numba_func(test_func)

    def test_assert_not_none(self):
        checks.assert_not_none(0)
        with pytest.raises(Exception):
            checks.assert_not_none(None)

    def test_assert_type(self):
        checks.assert_instance_of(0, int)
        checks.assert_instance_of(np.zeros(1), (np.ndarray, pd.Series))
        checks.assert_instance_of(pd.Series([1, 2, 3]), (np.ndarray, pd.Series))
        with pytest.raises(Exception):
            checks.assert_instance_of(pd.DataFrame([1, 2, 3]), (np.ndarray, pd.Series))

    def test_assert_subclass_of(self):
        class A:
            pass

        class B(A):
            pass

        class C(B):
            pass

        checks.assert_subclass_of(B, A)
        checks.assert_subclass_of(C, B)
        checks.assert_subclass_of(C, A)
        with pytest.raises(Exception):
            checks.assert_subclass_of(A, B)

    def test_assert_type_equal(self):
        checks.assert_type_equal(0, 1)
        checks.assert_type_equal(np.zeros(1), np.empty(1))
        with pytest.raises(Exception):
            checks.assert_instance_of(0, np.zeros(1))

    def test_assert_dtype(self):
        checks.assert_dtype(np.zeros(1), np.float_)
        checks.assert_dtype(pd.Series([1, 2, 3]), np.int_)
        checks.assert_dtype(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), np.int_)
        with pytest.raises(Exception):
            checks.assert_dtype(pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}), np.int_)

    def test_assert_subdtype(self):
        checks.assert_subdtype([0], np.number)
        checks.assert_subdtype(np.array([1, 2, 3]), np.number)
        checks.assert_subdtype(pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}), np.number)
        with pytest.raises(Exception):
            checks.assert_subdtype(np.array([1, 2, 3]), np.floating)
        with pytest.raises(Exception):
            checks.assert_subdtype(pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}), np.floating)

    def test_assert_dtype_equal(self):
        checks.assert_dtype_equal([1], [1, 1, 1])
        checks.assert_dtype_equal(pd.Series([1, 2, 3]), pd.DataFrame([[1, 2, 3]]))
        checks.assert_dtype_equal(pd.DataFrame([[1, 2, 3.0]]), pd.DataFrame([[1, 2, 3.0]]))
        with pytest.raises(Exception):
            checks.assert_dtype_equal(pd.DataFrame([[1, 2, 3]]), pd.DataFrame([[1, 2, 3.0]]))

    def test_assert_ndim(self):
        checks.assert_ndim(0, 0)
        checks.assert_ndim(np.zeros(1), 1)
        checks.assert_ndim(pd.Series([1, 2, 3]), (1, 2))
        checks.assert_ndim(pd.DataFrame([1, 2, 3]), (1, 2))
        with pytest.raises(Exception):
            checks.assert_ndim(np.zeros((3, 3, 3)), (1, 2))

    def test_assert_len_equal(self):
        checks.assert_len_equal([[1]], [[2]])
        checks.assert_len_equal([[1]], [[2, 3]])
        with pytest.raises(Exception):
            checks.assert_len_equal([[1]], [[2], [3]])

    def test_assert_shape_equal(self):
        checks.assert_shape_equal(0, 1)
        checks.assert_shape_equal([1, 2, 3], np.array([1, 2, 3]))
        checks.assert_shape_equal([1, 2, 3], pd.Series([1, 2, 3]))
        checks.assert_shape_equal(np.zeros((3, 3)), pd.Series([1, 2, 3]), axis=0)
        checks.assert_shape_equal(np.zeros((2, 3)), pd.Series([1, 2, 3]), axis=(1, 0))
        with pytest.raises(Exception):
            checks.assert_shape_equal(np.zeros((2, 3)), pd.Series([1, 2, 3]), axis=(0, 1))

    def test_assert_index_equal(self):
        checks.assert_index_equal(pd.Index([1, 2, 3]), pd.Index([1, 2, 3]))
        with pytest.raises(Exception):
            checks.assert_index_equal(pd.Index([1, 2, 3]), pd.Index([2, 3, 4]))

    def test_assert_meta_equal(self):
        index = ["x", "y", "z"]
        columns = ["a", "b", "c"]
        checks.assert_meta_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
        checks.assert_meta_equal(pd.Series([1, 2, 3], index=index), pd.Series([1, 2, 3], index=index))
        checks.assert_meta_equal(pd.DataFrame([[1, 2, 3]], columns=columns), pd.DataFrame([[1, 2, 3]], columns=columns))
        with pytest.raises(Exception):
            checks.assert_meta_equal(pd.Series([1, 2]), pd.DataFrame([1, 2]))

        with pytest.raises(Exception):
            checks.assert_meta_equal(pd.DataFrame([1, 2]), pd.DataFrame([1, 2, 3]))

        with pytest.raises(Exception):
            checks.assert_meta_equal(pd.DataFrame([1, 2, 3]), pd.DataFrame([1, 2, 3], index=index))

        with pytest.raises(Exception):
            checks.assert_meta_equal(pd.DataFrame([[1, 2, 3]]), pd.DataFrame([[1, 2, 3]], columns=columns))

    def test_assert_array_equal(self):
        index = ["x", "y", "z"]
        columns = ["a", "b", "c"]
        checks.assert_array_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
        checks.assert_array_equal(pd.Series([1, 2, 3], index=index), pd.Series([1, 2, 3], index=index))
        checks.assert_array_equal(
            pd.DataFrame([[1, 2, 3]], columns=columns),
            pd.DataFrame([[1, 2, 3]], columns=columns),
        )
        with pytest.raises(Exception):
            checks.assert_array_equal(np.array([1, 2]), np.array([1, 2, 3]))

    def test_assert_level_not_exists(self):
        i = pd.Index(["x", "y", "z"], name="i")
        multi_i = pd.MultiIndex.from_arrays([["x", "y", "z"], ["x2", "y2", "z2"]], names=["i", "i2"])
        checks.assert_level_not_exists(i, "i2")
        checks.assert_level_not_exists(multi_i, "i3")
        with pytest.raises(Exception):
            checks.assert_level_not_exists(i, "i")
            checks.assert_level_not_exists(multi_i, "i")

    def test_assert_equal(self):
        checks.assert_equal(0, 0)
        checks.assert_equal(False, False)
        with pytest.raises(Exception):
            checks.assert_equal(0, 1)

    def test_assert_dict_valid(self):
        checks.assert_dict_valid(dict(a=2, b=3), [["a", "b", "c"]])
        with pytest.raises(Exception):
            checks.assert_dict_valid(dict(a=2, b=3, d=4), [["a", "b", "c"]])
        checks.assert_dict_valid(dict(a=2, b=3, c=dict(d=4, e=5)), [["a", "b", "c"], ["d", "e"]])
        with pytest.raises(Exception):
            checks.assert_dict_valid(dict(a=2, b=3, c=dict(d=4, f=5)), [["a", "b", "c"], ["d", "e"]])


# ############# math_ ############# #


class TestMath:
    def test_is_close(self):
        a = 0.3
        b = 0.1 + 0.2

        # test scalar
        assert math_.is_close_nb(a, a)
        assert math_.is_close_nb(a, b)
        assert math_.is_close_nb(-a, -b)
        assert not math_.is_close_nb(-a, b)
        assert not math_.is_close_nb(a, -b)
        assert math_.is_close_nb(1e10 + a, 1e10 + b)

        # test np.nan
        assert not math_.is_close_nb(np.nan, b)
        assert not math_.is_close_nb(a, np.nan)

        # test np.inf
        assert not math_.is_close_nb(np.inf, b)
        assert not math_.is_close_nb(a, np.inf)
        assert not math_.is_close_nb(-np.inf, b)
        assert not math_.is_close_nb(a, -np.inf)
        assert not math_.is_close_nb(-np.inf, -np.inf)
        assert not math_.is_close_nb(np.inf, np.inf)
        assert not math_.is_close_nb(-np.inf, np.inf)

    def test_is_close_or_less(self):
        a = 0.3
        b = 0.1 + 0.2

        # test scalar
        assert math_.is_close_or_less_nb(a, a)
        assert math_.is_close_or_less_nb(a, b)
        assert math_.is_close_or_less_nb(-a, -b)
        assert math_.is_close_or_less_nb(-a, b)
        assert not math_.is_close_or_less_nb(a, -b)
        assert math_.is_close_or_less_nb(1e10 + a, 1e10 + b)

        # test np.nan
        assert not math_.is_close_or_less_nb(np.nan, b)
        assert not math_.is_close_or_less_nb(a, np.nan)

        # test np.inf
        assert not math_.is_close_or_less_nb(np.inf, b)
        assert math_.is_close_or_less_nb(a, np.inf)
        assert math_.is_close_or_less_nb(-np.inf, b)
        assert not math_.is_close_or_less_nb(a, -np.inf)
        assert not math_.is_close_or_less_nb(-np.inf, -np.inf)
        assert not math_.is_close_or_less_nb(np.inf, np.inf)
        assert math_.is_close_or_less_nb(-np.inf, np.inf)

    def test_is_less(self):
        a = 0.3
        b = 0.1 + 0.2

        # test scalar
        assert not math_.is_less_nb(a, a)
        assert not math_.is_less_nb(a, b)
        assert not math_.is_less_nb(-a, -b)
        assert math_.is_less_nb(-a, b)
        assert not math_.is_less_nb(a, -b)
        assert not math_.is_less_nb(1e10 + a, 1e10 + b)

        # test np.nan
        assert not math_.is_less_nb(np.nan, b)
        assert not math_.is_less_nb(a, np.nan)

        # test np.inf
        assert not math_.is_less_nb(np.inf, b)
        assert math_.is_less_nb(a, np.inf)
        assert math_.is_less_nb(-np.inf, b)
        assert not math_.is_less_nb(a, -np.inf)
        assert not math_.is_less_nb(-np.inf, -np.inf)
        assert not math_.is_less_nb(np.inf, np.inf)
        assert math_.is_less_nb(-np.inf, np.inf)

    def test_is_addition_zero(self):
        a = 0.3
        b = 0.1 + 0.2

        assert not math_.is_addition_zero_nb(a, b)
        assert math_.is_addition_zero_nb(-a, b)
        assert math_.is_addition_zero_nb(a, -b)
        assert not math_.is_addition_zero_nb(-a, -b)

    def test_add_nb(self):
        a = 0.3
        b = 0.1 + 0.2

        assert math_.add_nb(a, b) == a + b
        assert math_.add_nb(-a, b) == 0
        assert math_.add_nb(a, -b) == 0
        assert math_.add_nb(-a, -b) == -(a + b)


# ############# array_ ############# #


class TestArray:
    def test_is_sorted(self):
        assert array_.is_sorted(np.array([0, 1, 2, 3, 4]))
        assert array_.is_sorted(np.array([0, 1]))
        assert array_.is_sorted(np.array([0]))
        assert not array_.is_sorted(np.array([1, 0]))
        assert not array_.is_sorted(np.array([0, 1, 2, 4, 3]))
        # nb
        assert array_.is_sorted_nb(np.array([0, 1, 2, 3, 4]))
        assert array_.is_sorted_nb(np.array([0, 1]))
        assert array_.is_sorted_nb(np.array([0]))
        assert not array_.is_sorted_nb(np.array([1, 0]))
        assert not array_.is_sorted_nb(np.array([0, 1, 2, 4, 3]))

    def test_insert_argsort_nb(self):
        a = np.random.uniform(size=1000)
        A = a.copy()
        I = np.arange(len(A))
        array_.insert_argsort_nb(A, I)
        np.testing.assert_array_equal(np.sort(a), A)
        np.testing.assert_array_equal(a[I], A)

    def test_get_ranges_arr(self):
        np.testing.assert_array_equal(array_.get_ranges_arr(0, 3), np.array([0, 1, 2]))
        np.testing.assert_array_equal(array_.get_ranges_arr(0, [1, 2, 3]), np.array([0, 0, 1, 0, 1, 2]))
        np.testing.assert_array_equal(array_.get_ranges_arr([0, 3], [3, 6]), np.array([0, 1, 2, 3, 4, 5]))

    def test_uniform_summing_to_one_nb(self):
        @njit
        def set_seed():
            np.random.seed(seed)

        set_seed()
        np.testing.assert_array_almost_equal(
            array_.uniform_summing_to_one_nb(10),
            np.array(
                [
                    5.808361e-02,
                    9.791091e-02,
                    2.412011e-05,
                    2.185215e-01,
                    2.241184e-01,
                    2.456528e-03,
                    1.308789e-01,
                    1.341822e-01,
                    8.453816e-02,
                    4.928569e-02,
                ]
            ),
        )
        assert np.sum(array_.uniform_summing_to_one_nb(10)) == 1

    def test_rescale(self):
        assert array_.rescale(0, (0, 10), (0, 1)) == 0
        assert array_.rescale(10, (0, 10), (0, 1)) == 1
        np.testing.assert_array_equal(
            array_.rescale(np.array([0, 2, 4, 6, 8, 10]), (0, 10), (0, 1)),
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        )
        np.testing.assert_array_equal(
            array_.rescale_nb(np.array([0, 2, 4, 6, 8, 10]), (0, 10), (0, 1)),
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        )

    def test_min_rel_rescale(self):
        np.testing.assert_array_equal(
            array_.min_rel_rescale(np.array([2, 4, 6]), (10, 20)),
            np.array([10.0, 15.0, 20.0]),
        )
        np.testing.assert_array_equal(
            array_.min_rel_rescale(np.array([5, 6, 7]), (10, 20)),
            np.array([10.0, 12.0, 14.0]),
        )
        np.testing.assert_array_equal(
            array_.min_rel_rescale(np.array([5, 5, 5]), (10, 20)),
            np.array([10.0, 10.0, 10.0]),
        )

    def test_max_rel_rescale(self):
        np.testing.assert_array_equal(
            array_.max_rel_rescale(np.array([2, 4, 6]), (10, 20)),
            np.array([10.0, 15.0, 20.0]),
        )
        np.testing.assert_array_equal(
            array_.max_rel_rescale(np.array([5, 6, 7]), (10, 20)),
            np.array([14.285714285714286, 17.142857142857142, 20.0]),
        )
        np.testing.assert_array_equal(
            array_.max_rel_rescale(np.array([5, 5, 5]), (10, 20)),
            np.array([20.0, 20.0, 20.0]),
        )

    def test_rescale_float_to_int_nb(self):
        @njit
        def set_seed():
            np.random.seed(seed)

        set_seed()
        np.testing.assert_array_equal(
            array_.rescale_float_to_int_nb(np.array([0.3, 0.3, 0.3, 0.1]), (10, 20), 70),
            np.array([17, 14, 22, 17]),
        )
        assert np.sum(array_.rescale_float_to_int_nb(np.array([0.3, 0.3, 0.3, 0.1]), (10, 20), 70)) == 70


# ############# random_ ############# #


class TestRandom:
    def test_set_seed(self):
        random_.set_seed(seed)

        def test_seed():
            return np.random.uniform(0, 1)

        assert test_seed() == 0.3745401188473625

        if "NUMBA_DISABLE_JIT" not in os.environ or os.environ["NUMBA_DISABLE_JIT"] != "1":

            @njit
            def test_seed_nb():
                return np.random.uniform(0, 1)

            assert test_seed_nb() == 0.3745401188473625


# ############# mapping ############# #

Enum = namedtuple("Enum", ["Attr1", "Attr2"])(*range(2))


class TestMapping:
    def test_to_value_mapping(self):
        assert mapping.to_value_mapping(Enum) == {0: "Attr1", 1: "Attr2", -1: None}
        assert mapping.to_value_mapping(Enum, reverse=True) == {"Attr1": 0, "Attr2": 1, None: -1}
        assert mapping.to_value_mapping({0: "Attr1", 1: "Attr2", -1: None}) == {0: "Attr1", 1: "Attr2", -1: None}
        assert mapping.to_value_mapping(["Attr1", "Attr2"]) == {0: "Attr1", 1: "Attr2"}
        assert mapping.to_value_mapping(pd.Index(["Attr1", "Attr2"])) == {0: "Attr1", 1: "Attr2"}
        assert mapping.to_value_mapping(pd.Series(["Attr1", "Attr2"])) == {0: "Attr1", 1: "Attr2"}

    def test_apply_mapping(self):
        assert mapping.apply_mapping("Attr1", mapping_like=Enum, reverse=True) == 0
        with pytest.raises(Exception):
            mapping.apply_mapping("Attr3", mapping_like=Enum, reverse=True)
        assert mapping.apply_mapping("attr1", mapping_like=Enum, reverse=True, ignore_case=True) == 0
        with pytest.raises(Exception):
            mapping.apply_mapping("attr1", mapping_like=Enum, reverse=True, ignore_case=False)
        assert mapping.apply_mapping("Attr_1", mapping_like=Enum, reverse=True, ignore_underscores=True) == 0
        with pytest.raises(Exception):
            mapping.apply_mapping("Attr_1", mapping_like=Enum, reverse=True, ignore_underscores=False)
        assert (
            mapping.apply_mapping("attr_1", mapping_like=Enum, reverse=True, ignore_case=True, ignore_underscores=True)
            == 0
        )
        with pytest.raises(Exception):
            mapping.apply_mapping("attr_1", mapping_like=Enum, reverse=True, ignore_case=True, ignore_underscores=False)
        assert mapping.apply_mapping(np.array([1]), mapping_like={1: "hello"})[0] == "hello"
        assert mapping.apply_mapping(np.array([1]), mapping_like={1.0: "hello"})[0] == "hello"
        assert mapping.apply_mapping(np.array([1.0]), mapping_like={1: "hello"})[0] == "hello"
        assert mapping.apply_mapping(np.array([True]), mapping_like={1: "hello"})[0] == "hello"
        assert mapping.apply_mapping(np.array([True]), mapping_like={True: "hello"})[0] == "hello"
        with pytest.raises(Exception):
            mapping.apply_mapping(np.array([True]), mapping_like={"world": "hello"})
        with pytest.raises(Exception):
            mapping.apply_mapping(np.array([1]), mapping_like={"world": "hello"})
        assert mapping.apply_mapping(np.array(["world"]), mapping_like={"world": "hello"})[0] == "hello"


# ############# enum_ ############# #


class TestEnum:
    def test_map_enum_fields(self):
        assert enum_.map_enum_fields(0, Enum) == 0
        assert enum_.map_enum_fields(10, Enum) == 10
        with pytest.raises(Exception):
            enum_.map_enum_fields(10.0, Enum)
        assert enum_.map_enum_fields("Attr1", Enum) == 0
        assert enum_.map_enum_fields("attr1", Enum) == 0
        with pytest.raises(Exception):
            enum_.map_enum_fields("hello", Enum)
        assert enum_.map_enum_fields("attr1", Enum) == 0
        assert enum_.map_enum_fields(("attr1", "attr2"), Enum) == (0, 1)
        assert enum_.map_enum_fields([["attr1", "attr2"]], Enum) == [[0, 1]]
        np.testing.assert_array_equal(enum_.map_enum_fields(np.array([]), Enum), np.array([]))
        with pytest.raises(Exception):
            enum_.map_enum_fields(np.array([[0.0, 1.0]]), Enum)
        with pytest.raises(Exception):
            enum_.map_enum_fields(np.array([[False, True]]), Enum)
        np.testing.assert_array_equal(enum_.map_enum_fields(np.array([[0, 1]]), Enum), np.array([[0, 1]]))
        np.testing.assert_array_equal(enum_.map_enum_fields(np.array([["attr1", "attr2"]]), Enum), np.array([[0, 1]]))
        with pytest.raises(Exception):
            enum_.map_enum_fields(np.array([["attr1", 1]]), Enum)
        assert_series_equal(enum_.map_enum_fields(pd.Series([]), Enum), pd.Series([]))
        with pytest.raises(Exception):
            enum_.map_enum_fields(pd.Series([0.0, 1.0]), Enum)
        with pytest.raises(Exception):
            enum_.map_enum_fields(pd.Series([False, True]), Enum)
        assert_series_equal(enum_.map_enum_fields(pd.Series([0, 1]), Enum), pd.Series([0, 1]))
        assert_series_equal(enum_.map_enum_fields(pd.Series(["attr1", "attr2"]), Enum), pd.Series([0, 1]))
        with pytest.raises(Exception):
            enum_.map_enum_fields(pd.Series(["attr1", 0]), Enum)
        assert_frame_equal(enum_.map_enum_fields(pd.DataFrame([]), Enum), pd.DataFrame([]))
        with pytest.raises(Exception):
            enum_.map_enum_fields(pd.DataFrame([[0.0, 1.0]]), Enum)
        assert_frame_equal(enum_.map_enum_fields(pd.DataFrame([[0, 1]]), Enum), pd.DataFrame([[0, 1]]))
        assert_frame_equal(
            enum_.map_enum_fields(pd.DataFrame([["attr1", "attr2"]]), Enum),
            pd.DataFrame([[0, 1]]),
        )
        assert_frame_equal(enum_.map_enum_fields(pd.DataFrame([[0, "attr2"]]), Enum), pd.DataFrame([[0, 1]]))

    def test_map_enum_values(self):
        assert enum_.map_enum_values(0, Enum) == "Attr1"
        assert enum_.map_enum_values(-1, Enum) is None
        with pytest.raises(Exception):
            enum_.map_enum_values(-2, Enum)
        assert enum_.map_enum_values((0, 1, "Attr3"), Enum) == ("Attr1", "Attr2", "Attr3")
        assert enum_.map_enum_values([[0, 1, "Attr3"]], Enum) == [["Attr1", "Attr2", "Attr3"]]
        assert enum_.map_enum_values("hello", Enum) == "hello"
        np.testing.assert_array_equal(enum_.map_enum_values(np.array([]), Enum), np.array([]))
        np.testing.assert_array_equal(
            enum_.map_enum_values(np.array([[0.0, 1.0]]), Enum),
            np.array([["Attr1", "Attr2"]]),
        )
        np.testing.assert_array_equal(
            enum_.map_enum_values(np.array([["Attr1", "Attr2"]]), Enum),
            np.array([["Attr1", "Attr2"]]),
        )
        np.testing.assert_array_equal(enum_.map_enum_values(np.array([[0, "Attr2"]]), Enum), np.array([["0", "Attr2"]]))
        assert_series_equal(enum_.map_enum_values(pd.Series([]), Enum), pd.Series([]))
        assert_series_equal(
            enum_.map_enum_values(pd.Series([0.0, 1.0]), Enum),
            pd.Series(["Attr1", "Attr2"]),
        )
        assert_series_equal(enum_.map_enum_values(pd.Series([0, 1]), Enum), pd.Series(["Attr1", "Attr2"]))
        assert_series_equal(
            enum_.map_enum_values(pd.Series(["Attr1", "Attr2"]), Enum),
            pd.Series(["Attr1", "Attr2"]),
        )
        with pytest.raises(Exception):
            enum_.map_enum_values(pd.Series([0, "Attr2"]), Enum)
        assert_frame_equal(enum_.map_enum_values(pd.DataFrame([]), Enum), pd.DataFrame([]))
        assert_frame_equal(
            enum_.map_enum_values(pd.DataFrame([[0.0, 1.0]]), Enum),
            pd.DataFrame([["Attr1", "Attr2"]]),
        )
        assert_frame_equal(
            enum_.map_enum_values(pd.DataFrame([[0, 1]]), Enum),
            pd.DataFrame([["Attr1", "Attr2"]]),
        )
        assert_frame_equal(
            enum_.map_enum_values(pd.DataFrame([["Attr1", "Attr2"]]), Enum),
            pd.DataFrame([["Attr1", "Attr2"]]),
        )
        assert_frame_equal(
            enum_.map_enum_values(pd.DataFrame([[0, "Attr2"]]), Enum),
            pd.DataFrame([["Attr1", "Attr2"]]),
        )


# ############# params ############# #


class TestParams:
    def test_create_param_combs(self):
        assert params.generate_param_combs((combinations, [0, 1, 2, 3], 2)) == [[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]]
        assert params.generate_param_combs((product, (combinations, [0, 1, 2, 3], 2), [4, 5])) == [
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
            [1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3],
            [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5],
        ]
        assert params.generate_param_combs((product, (combinations, [0, 1, 2], 2), (combinations, [3, 4, 5], 2))) == [
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 2, 2, 2, 2, 2, 2],
            [3, 3, 4, 3, 3, 4, 3, 3, 4],
            [4, 5, 5, 4, 5, 5, 4, 5, 5],
        ]

    def test_find_params(self):
        assert params.find_params_in_obj(
            {
                "a": 1,
                "b": params.Param([1, 2, 3]),
                "c": {"d": 2, "e": params.Param([1, 2, 3]), "f": (3, params.Param([1, 2, 3]))},
            }
        ) == {
            "b": params.Param([1, 2, 3]),
            ("c", "e"): params.Param([1, 2, 3]),
            ("c", "f", 1): params.Param([1, 2, 3]),
        }

    def test_parameterized(self):
        def f(a, *my_args, b=2, **my_kwargs):
            return a, my_args, b, my_kwargs

        def merge_func(results, param_index):
            return results, param_index

        fp = params.parameterized(f, merge_func=merge_func, merge_kwargs=dict(param_index=template.Rep("param_index")))
        assert fp(1) == (1, (), 2, {})
        assert fp(1, 2) == (1, (2,), 2, {})
        assert fp(1, 2, 3) == (1, (2, 3), 2, {})
        assert fp(1, 2, 3, b=4) == (1, (2, 3), 4, {})
        assert fp(1, 2, 3, b=4, c=5) == (1, (2, 3), 4, {"c": 5})

        assert fp(vbt.Param([1]))[0] == [(1, (), 2, {})]
        assert_index_equal(fp(vbt.Param([1]))[1], pd.Index([1], dtype="int64", name="a"))
        assert fp(vbt.Param([1, 2]))[0] == [(1, (), 2, {}), (2, (), 2, {})]
        assert_index_equal(fp(vbt.Param([1, 2]))[1], pd.Index([1, 2], dtype="int64", name="a"))
        assert fp(1, vbt.Param([2, 3]))[0] == [(1, (2,), 2, {}), (1, (3,), 2, {})]
        assert_index_equal(fp(1, vbt.Param([2, 3]))[1], pd.Index([2, 3], dtype="int64", name="my_args_0"))
        assert fp(1, b=vbt.Param([2, 3]))[0] == [(1, (), 2, {}), (1, (), 3, {})]
        assert_index_equal(fp(1, b=vbt.Param([2, 3]))[1], pd.Index([2, 3], dtype="int64", name="b"))
        assert fp(1, c=vbt.Param([2, 3]))[0] == [(1, (), 2, {"c": 2}), (1, (), 2, {"c": 3})]
        assert_index_equal(fp(1, c=vbt.Param([2, 3]))[1], pd.Index([2, 3], dtype="int64", name="c"))
        kwargs = dict(c=dict(d=(2, dict(e=vbt.Param([3, 4])))), f=(5, vbt.Param([6, 7])))
        assert fp(1, **kwargs)[0] == [
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 6)}),
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)}),
            (1, (), 2, {"c": dict(d=(2, dict(e=4))), "f": (5, 6)}),
            (1, (), 2, {"c": dict(d=(2, dict(e=4))), "f": (5, 7)}),
        ]
        assert_index_equal(
            fp(1, **kwargs)[1],
            pd.MultiIndex.from_tuples(
                [
                    (3, 6),
                    (3, 7),
                    (4, 6),
                    (4, 7),
                ],
                names=["c_d_1_e", "f_1"],
            ),
        )
        assert fp(1, **kwargs, _selection=1) == (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)})
        assert fp(1, **kwargs, _selection=1, _skip_single_param=False)[0] == [
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)}),
        ]
        assert_index_equal(
            fp(1, **kwargs, _selection=1, _skip_single_param=False)[1],
            pd.MultiIndex.from_tuples(
                [
                    (3, 7),
                ],
                names=["c_d_1_e", "f_1"],
            ),
        )
        assert fp(1, **kwargs, _selection=(3, 7)) == (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)})
        assert fp(1, **kwargs, _selection=(3, 7), _skip_single_param=False)[0] == [
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)}),
        ]
        assert_index_equal(
            fp(1, **kwargs, _selection=(3, 7), _skip_single_param=False)[1],
            pd.MultiIndex.from_tuples(
                [
                    (3, 7),
                ],
                names=["c_d_1_e", "f_1"],
            ),
        )
        assert fp(1, **kwargs, _selection=[1])[0] == [
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)}),
        ]
        assert_index_equal(
            fp(1, **kwargs, _selection=[1])[1],
            pd.MultiIndex.from_tuples(
                [
                    (3, 7),
                ],
                names=["c_d_1_e", "f_1"],
            ),
        )
        assert fp(1, **kwargs, _selection=[1, (4, 7)])[0] == [
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)}),
            (1, (), 2, {"c": dict(d=(2, dict(e=4))), "f": (5, 7)}),
        ]
        assert_index_equal(
            fp(1, **kwargs, _selection=[1, (4, 7)])[1],
            pd.MultiIndex.from_tuples(
                [
                    (3, 7),
                    (4, 7),
                ],
                names=["c_d_1_e", "f_1"],
            ),
        )
        assert fp(1, **kwargs, _selection=vbt.RepFunc(lambda param_index: param_index[[1, 3]]))[0] == [
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)}),
            (1, (), 2, {"c": dict(d=(2, dict(e=4))), "f": (5, 7)}),
        ]
        assert_index_equal(
            fp(1, **kwargs, _selection=vbt.RepFunc(lambda param_index: param_index[[1, 3]]))[1],
            pd.MultiIndex.from_tuples(
                [
                    (3, 7),
                    (4, 7),
                ],
                names=["c_d_1_e", "f_1"],
            ),
        )

        param_configs = [dict(a=1)]
        assert fp(param_configs=param_configs)[0] == [(1, (), 2, {})]
        assert fp(param_configs=param_configs)[1] is None
        param_configs = [dict(a=1, my_args=(2, 3))]
        assert fp(param_configs=param_configs)[0] == [(1, (2, 3), 2, {})]
        assert fp(param_configs=param_configs)[1] is None
        param_configs = [dict(a=1, my_args_0=2, my_args_1=3)]
        assert fp(param_configs=param_configs)[0] == [(1, (2, 3), 2, {})]
        assert fp(param_configs=param_configs)[1] is None
        param_configs = [dict(a=1, b=3)]
        assert fp(param_configs=param_configs)[0] == [(1, (), 3, {})]
        assert fp(param_configs=param_configs)[1] is None
        param_configs = [dict(a=1, my_kwargs=dict(c=3))]
        assert fp(param_configs=param_configs)[0] == [(1, (), 2, {"c": 3})]
        assert fp(param_configs=param_configs)[1] is None
        param_configs = [dict(a=1, c=3)]
        assert fp(param_configs=param_configs)[0] == [(1, (), 2, {"c": 3})]
        assert fp(param_configs=param_configs)[1] is None
        param_configs = [dict(a=2, my_args=(3, 4)), dict(b=5, my_kwargs=dict(c=6))]
        assert fp(1, 1, 1, param_configs=param_configs)[0] == [(2, (3, 4), 2, {}), (1, (1, 1), 5, {"c": 6})]
        assert_index_equal(
            fp(1, 1, 1, param_configs=param_configs)[1],
            pd.Index([0, 1], dtype="int64", name="param_config"),
        )

        param_configs = [dict(b=3)]
        assert fp(vbt.Param([2]), param_configs=param_configs)[0] == [
            (2, (), 3, {}),
        ]
        assert_index_equal(
            fp(vbt.Param([2]), param_configs=param_configs)[1],
            pd.Index([2], dtype="int64", name="a"),
        )
        param_configs = [dict(b=3, _name="my_config")]
        assert fp(vbt.Param([2]), param_configs=param_configs)[0] == [
            (2, (), 3, {}),
        ]
        assert_index_equal(
            fp(vbt.Param([2]), param_configs=param_configs)[1],
            pd.MultiIndex.from_tuples(
                [
                    (2, "my_config"),
                ],
                names=["a", "param_config"],
            ),
        )
        param_configs = [dict(b=3), dict(b=4)]
        assert fp(vbt.Param([1, 2]), param_configs=param_configs)[0] == [
            (1, (), 3, {}),
            (1, (), 4, {}),
            (2, (), 3, {}),
            (2, (), 4, {}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs)[1],
            pd.MultiIndex.from_tuples([(1, 0), (1, 1), (2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[0] == [
            ([1, 1], (), [3, 4], {}),
            ([2, 2], (), [3, 4], {}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][0],
            pd.MultiIndex.from_tuples([(1, 0), (1, 1)], names=["a", "param_config"]),
        )
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][1],
            pd.MultiIndex.from_tuples([(2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert fp(vbt.Param([1, 2], mono_reduce=True), param_configs=param_configs, _mono_chunk_len=2)[0] == [
            (1, (), [3, 4], {}),
            (2, (), [3, 4], {}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][0],
            pd.MultiIndex.from_tuples([(1, 0), (1, 1)], names=["a", "param_config"]),
        )
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][1],
            pd.MultiIndex.from_tuples([(2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert fp(vbt.Param([1, 2]), hello="world", param_configs=param_configs, _mono_chunk_len=2)[0] == [
            ([1, 1], (), [3, 4], {"hello": "world"}),
            ([2, 2], (), [3, 4], {"hello": "world"}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][0],
            pd.MultiIndex.from_tuples([(1, 0), (1, 1)], names=["a", "param_config"]),
        )
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][1],
            pd.MultiIndex.from_tuples([(2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert fp(
            vbt.Param([1, 2]),
            hello="world",
            param_configs=param_configs,
            _mono_chunk_len=2,
            _mono_reduce=dict(hello=False),
        )[0] == [
            ([1, 1], (), [3, 4], {"hello": ["world", "world"]}),
            ([2, 2], (), [3, 4], {"hello": ["world", "world"]}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][0],
            pd.MultiIndex.from_tuples([(1, 0), (1, 1)], names=["a", "param_config"]),
        )
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][1],
            pd.MultiIndex.from_tuples([(2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert fp(
            vbt.Param([1, 2]),
            hello="world",
            param_configs=param_configs,
            _mono_chunk_len=2,
            _mono_reduce=True,
        )[0] == [
            (1, (), [3, 4], {"hello": "world"}),
            (2, (), [3, 4], {"hello": "world"}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][0],
            pd.MultiIndex.from_tuples([(1, 0), (1, 1)], names=["a", "param_config"]),
        )
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][1],
            pd.MultiIndex.from_tuples([(2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert fp(
            vbt.Param([1, 2], mono_merge_func=sum),
            param_configs=param_configs,
            _mono_chunk_len=2,
            _mono_merge_func=dict(b=sum),
        )[0] == [
            (2, (), 7, {}),
            (4, (), 7, {}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][0],
            pd.MultiIndex.from_tuples([(1, 0), (1, 1)], names=["a", "param_config"]),
        )
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][1],
            pd.MultiIndex.from_tuples([(2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert fp(
            vbt.Param([1, 2]),
            param_configs=param_configs,
            _mono_chunk_len=2,
            _mono_merge_func=sum,
        )[0] == fp(
            vbt.Param([1, 2], mono_merge_func=sum),
            param_configs=param_configs,
            _mono_chunk_len=2,
            _mono_merge_func=dict(b=sum),
        )[0]

# ############# datetime_ ############# #


class TestDatetime:
    def test_to_timedelta(self):
        assert datetime_.freq_to_timedelta("d") == pd.to_timedelta("1d")
        assert datetime_.freq_to_timedelta("day") == pd.to_timedelta("1d")
        assert datetime_.freq_to_timedelta("m") == pd.to_timedelta("1min")
        assert datetime_.freq_to_timedelta("1m") == pd.to_timedelta("1min")
        assert datetime_.freq_to_timedelta("1 m") == pd.to_timedelta("1min")
        assert datetime_.freq_to_timedelta("1 minute") == pd.to_timedelta("1min")
        assert datetime_.freq_to_timedelta("2 minutes") == pd.to_timedelta("2min")
        with pytest.raises(Exception):
            datetime_.freq_to_timedelta("1")

    def test_get_utc_tz(self):
        assert datetime_.get_utc_tz().utcoffset(_datetime.now()) == _timedelta(0)

    def test_get_local_tz(self):
        assert datetime_.get_local_tz().utcoffset(_datetime.now()) == _datetime.now().astimezone(None).utcoffset()

    def test_convert_tzaware_time(self):
        assert datetime_.convert_tzaware_time(
            _time(12, 0, 0, tzinfo=datetime_.get_utc_tz()),
            _timezone(_timedelta(hours=2)),
        ) == _time(14, 0, 0, tzinfo=_timezone(_timedelta(hours=2)))

    def test_tzaware_to_naive_time(self):
        assert datetime_.tzaware_to_naive_time(
            _time(12, 0, 0, tzinfo=datetime_.get_utc_tz()),
            _timezone(_timedelta(hours=2)),
        ) == _time(14, 0, 0)

    def test_naive_to_tzaware_time(self):
        assert datetime_.naive_to_tzaware_time(
            _time(12, 0, 0),
            _timezone(_timedelta(hours=2)),
        ) == datetime_.convert_tzaware_time(
            _time(12, 0, 0, tzinfo=datetime_.get_local_tz()),
            _timezone(_timedelta(hours=2)),
        )

    def test_convert_naive_time(self):
        assert datetime_.convert_naive_time(
            _time(12, 0, 0),
            _timezone(_timedelta(hours=2)),
        ) == datetime_.tzaware_to_naive_time(
            _time(12, 0, 0, tzinfo=datetime_.get_local_tz()),
            _timezone(_timedelta(hours=2)),
        )

    def test_is_tz_aware(self):
        assert not datetime_.is_tz_aware(pd.Timestamp("2020-01-01"))
        assert datetime_.is_tz_aware(pd.Timestamp("2020-01-01", tz=datetime_.get_utc_tz()))

    def test_to_timezone(self):
        assert datetime_.to_timezone("UTC", to_fixed_offset=True) == _timezone.utc
        assert isinstance(datetime_.to_timezone("Europe/Berlin", to_fixed_offset=True), _timezone)
        assert datetime_.to_timezone("+0500") == _timezone(_timedelta(hours=5))
        assert datetime_.to_timezone(_timezone(_timedelta(hours=1))) == _timezone(_timedelta(hours=1))
        assert datetime_.to_timezone(3600) == _timezone(_timedelta(hours=1))
        assert datetime_.to_timezone(1800) == _timezone(_timedelta(hours=0.5))
        with pytest.raises(Exception):
            datetime_.to_timezone("+05")

    def test_to_tzaware_datetime(self):
        assert datetime_.to_tzaware_datetime(0) == _datetime(1970, 1, 1, 0, 0, 0, 0, tzinfo=datetime_.get_utc_tz())
        assert datetime_.to_tzaware_datetime(pd.Timestamp("2020-01-01").value, unit="ns") == _datetime(
            2020, 1, 1
        ).replace(tzinfo=datetime_.get_utc_tz())
        assert datetime_.to_tzaware_datetime("2020-01-01") == _datetime(2020, 1, 1).replace(
            tzinfo=datetime_.get_local_tz()
        )
        assert datetime_.to_tzaware_datetime(pd.Timestamp("2020-01-01")) == _datetime(2020, 1, 1).replace(
            tzinfo=datetime_.get_local_tz()
        )
        assert datetime_.to_tzaware_datetime(pd.Timestamp("2020-01-01", tz=datetime_.get_utc_tz())) == _datetime(
            2020,
            1,
            1,
        ).replace(tzinfo=datetime_.get_utc_tz())
        assert datetime_.to_tzaware_datetime(_datetime(2020, 1, 1)) == _datetime(2020, 1, 1).replace(
            tzinfo=datetime_.get_local_tz()
        )
        assert datetime_.to_tzaware_datetime(_datetime(2020, 1, 1, tzinfo=datetime_.get_utc_tz())) == _datetime(
            2020,
            1,
            1,
        ).replace(tzinfo=datetime_.get_utc_tz())
        assert datetime_.to_tzaware_datetime(
            _datetime(2020, 1, 1, 12, 0, 0, tzinfo=datetime_.get_utc_tz()),
            tz=datetime_.get_local_tz(),
        ) == _datetime(2020, 1, 1, 12, 0, 0, tzinfo=datetime_.get_utc_tz()).astimezone(datetime_.get_local_tz())
        with pytest.raises(Exception):
            datetime_.to_tzaware_datetime("2020-01-001")

    def test_datetime_to_ms(self):
        assert datetime_.datetime_to_ms(_datetime(2020, 1, 1, tzinfo=datetime_.get_utc_tz())) == 1577836800000


# ############# schedule_ ############# #


class TestScheduleManager:
    def test_every(self):
        manager = schedule_.ScheduleManager()
        job = manager.every()
        assert job.interval == 1
        assert job.unit == "seconds"
        assert job.at_time is None
        assert job.start_day is None

        job = manager.every(10, "minutes")
        assert job.interval == 10
        assert job.unit == "minutes"
        assert job.at_time is None
        assert job.start_day is None

        job = manager.every("hour")
        assert job.interval == 1
        assert job.unit == "hours"
        assert job.at_time is None
        assert job.start_day is None

        job = manager.every("10:30")
        assert job.interval == 1
        assert job.unit == "days"
        assert job.at_time == _time(10, 30)
        assert job.start_day is None

        job = manager.every("day", "10:30")
        assert job.interval == 1
        assert job.unit == "days"
        assert job.at_time == _time(10, 30)
        assert job.start_day is None

        job = manager.every("day", _time(9, 30, tzinfo=datetime_.get_utc_tz()))
        assert job.interval == 1
        assert job.unit == "days"
        assert job.at_time == datetime_.tzaware_to_naive_time(
            _time(9, 30, tzinfo=datetime_.get_utc_tz()),
            datetime_.get_local_tz(),
        )
        assert job.start_day is None

        job = manager.every("monday")
        assert job.interval == 1
        assert job.unit == "weeks"
        assert job.at_time is None
        assert job.start_day == "monday"

        job = manager.every("wednesday", "13:15")
        assert job.interval == 1
        assert job.unit == "weeks"
        assert job.at_time == _time(13, 15)
        assert job.start_day == "wednesday"

        job = manager.every("minute", ":17")
        assert job.interval == 1
        assert job.unit == "minutes"
        assert job.at_time == _time(0, 0, 17)
        assert job.start_day is None

    def test_start(self):
        kwargs = dict(call_count=0)

        def job_func(kwargs):
            kwargs["call_count"] += 1
            if kwargs["call_count"] == 5:
                raise KeyboardInterrupt

        manager = schedule_.ScheduleManager()
        manager.every().do(job_func, kwargs)
        manager.start()
        assert kwargs["call_count"] == 5

    def test_async_start(self):
        kwargs = dict(call_count=0)

        def job_func(kwargs):
            kwargs["call_count"] += 1
            if kwargs["call_count"] == 5:
                raise schedule_.CancelledError

        manager = schedule_.ScheduleManager()
        manager.every().do(job_func, kwargs)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(manager.async_start())
        assert kwargs["call_count"] == 5


# ############# tagging ############# #


class TestTags:
    def test_match_tags(self):
        assert tagging.match_tags("hello", "hello")
        assert not tagging.match_tags("hello", "world")
        assert tagging.match_tags(["hello", "world"], "world")
        assert tagging.match_tags("hello", ["hello", "world"])
        assert tagging.match_tags("hello and world", ["hello", "world"])
        assert not tagging.match_tags("hello and not world", ["hello", "world"])


# ############# template ############# #


class TestTemplate:
    def test_sub(self):
        assert template.Sub("$hello$world", {"hello": 100}).substitute({"world": 300}) == "100300"
        assert template.Sub("$hello$world", {"hello": 100}).substitute({"hello": 200, "world": 300}) == "200300"

    def test_rep(self):
        assert template.Rep("hello", {"hello": 100}).substitute() == 100
        assert template.Rep("hello", {"hello": 100}).substitute({"hello": 200}) == 200

    def test_repeval(self):
        assert template.RepEval("hello == 100", {"hello": 100}).substitute()
        assert not template.RepEval("hello == 100", {"hello": 100}).substitute({"hello": 200})

    def test_repfunc(self):
        assert template.RepFunc(lambda hello: hello == 100, {"hello": 100}).substitute()
        assert not template.RepFunc(lambda hello: hello == 100, {"hello": 100}).substitute({"hello": 200})

    def test_substitute_templates(self):
        assert template.substitute_templates(template.Rep("hello"), {"hello": 100}) == 100
        with pytest.raises(Exception):
            template.substitute_templates(template.Rep("hello2"), {"hello": 100})
        assert isinstance(
            template.substitute_templates(template.Rep("hello2"), {"hello": 100}, strict=False), template.Rep
        )
        assert template.substitute_templates(template.Sub("$hello"), {"hello": 100}) == "100"
        with pytest.raises(Exception):
            template.substitute_templates(template.Sub("$hello2"), {"hello": 100})
        assert template.substitute_templates([template.Rep("hello")], {"hello": 100}, except_types=()) == [100]
        assert template.substitute_templates({template.Rep("hello")}, {"hello": 100}, except_types=()) == {100}
        assert template.substitute_templates({"test": template.Rep("hello")}, {"hello": 100}) == {"test": 100}
        Tup = namedtuple("Tup", ["a"])
        tup = Tup(template.Rep("hello"))
        assert template.substitute_templates(tup, {"hello": 100}) == Tup(100)
        assert template.substitute_templates(template.RepEval("100"), max_depth=0) == 100
        assert template.substitute_templates((template.RepEval("100"),), max_depth=0) == (template.RepEval("100"),)
        assert template.substitute_templates((template.RepEval("100"),), max_depth=1) == (100,)
        assert template.substitute_templates((template.RepEval("100"),), max_len=1) == (100,)
        assert template.substitute_templates((0, template.RepEval("100")), max_len=1) == (
            0,
            template.RepEval("100"),
        )
        assert template.substitute_templates((0, template.RepEval("100")), max_len=2) == (
            0,
            100,
        )


# ############# parsing ############# #


class TestParsing:
    def test_get_func_kwargs(self):
        def f(a, *args, b=2, **kwargs):
            pass

        assert parsing.get_func_kwargs(f) == {"b": 2}

    def test_get_func_arg_names(self):
        def f(a, *args, b=2, **kwargs):
            pass

        assert parsing.get_func_arg_names(f) == ["a", "b"]

    def test_get_expr_var_names(self):
        assert parsing.get_expr_var_names("d = (a + b) / c") == ["d", "c", "a", "b"]

    def test_annotate_args(self):
        def f(a, *args, b=2, **kwargs):
            pass

        with pytest.raises(Exception):
            parsing.annotate_args(f, (), {})
        assert parsing.annotate_args(f, (1,), {}) == dict(
            a={"kind": inspect.Parameter.POSITIONAL_OR_KEYWORD, "value": 1},
            args={"kind": inspect.Parameter.VAR_POSITIONAL, "value": ()},
            b={"kind": inspect.Parameter.KEYWORD_ONLY, "value": 2},
            kwargs={"kind": inspect.Parameter.VAR_KEYWORD, "value": {}},
        )
        assert parsing.annotate_args(f, (1,), {}, only_passed=True) == dict(
            a={"kind": inspect.Parameter.POSITIONAL_OR_KEYWORD, "value": 1},
        )
        assert parsing.annotate_args(f, (1, 2, 3), {}) == dict(
            a={"kind": inspect.Parameter.POSITIONAL_OR_KEYWORD, "value": 1},
            args={"kind": inspect.Parameter.VAR_POSITIONAL, "value": (2, 3)},
            b={"kind": inspect.Parameter.KEYWORD_ONLY, "value": 2},
            kwargs={"kind": inspect.Parameter.VAR_KEYWORD, "value": {}},
        )

        def f2(a, b=2, **kwargs):
            pass

        assert parsing.annotate_args(f2, (1, 2), dict(c=3)) == dict(
            a={"kind": inspect.Parameter.POSITIONAL_OR_KEYWORD, "value": 1},
            b={"kind": inspect.Parameter.POSITIONAL_OR_KEYWORD, "value": 2},
            kwargs={"kind": inspect.Parameter.VAR_KEYWORD, "value": dict(c=3)},
        )

    def test_ann_args_to_args(self):
        def f(a, *args, b=2, **kwargs):
            pass

        assert parsing.ann_args_to_args(parsing.annotate_args(f, (1,), {})) == ((1,), {"b": 2})
        assert parsing.ann_args_to_args(parsing.annotate_args(f, (1,), {}, only_passed=True)) == ((1,), {})
        assert parsing.ann_args_to_args(parsing.annotate_args(f, (1, 2, 3), {})) == ((1, 2, 3), {"b": 2})

        def f2(a, b=2, **kwargs):
            pass

        assert parsing.ann_args_to_args(parsing.annotate_args(f2, (1, 2), dict(c=3))) == ((1, 2), {"c": 3})

    def test_match_ann_arg(self):
        def f(a, *args, b=2, **kwargs):
            pass

        with pytest.raises(Exception):
            parsing.annotate_args(f, (), {})

        ann_args = parsing.annotate_args(f, (0, 1), dict(c=3))

        assert parsing.match_ann_arg(ann_args, 0) == 0
        assert parsing.match_ann_arg(ann_args, "a") == 0
        assert parsing.match_ann_arg(ann_args, 1) == 1
        assert parsing.match_ann_arg(ann_args, 2) == 2
        assert parsing.match_ann_arg(ann_args, "b") == 2
        assert parsing.match_ann_arg(ann_args, parsing.Regex("(a|b)")) == 0
        assert parsing.match_ann_arg(ann_args, 3) == 3
        assert parsing.match_ann_arg(ann_args, "c") == 3
        with pytest.raises(Exception):
            parsing.match_ann_arg(ann_args, 4)
        with pytest.raises(Exception):
            parsing.match_ann_arg(ann_args, "d")

    def test_ignore_flat_ann_args(self):
        def f(a, *args, b=2, **kwargs):
            pass

        ann_args = parsing.annotate_args(f, (0, 1), dict(c=3))

        flat_ann_args = parsing.flatten_ann_args(ann_args)
        assert list(parsing.ignore_flat_ann_args(flat_ann_args, [0]).items()) == list(flat_ann_args.items())[1:]
        assert list(parsing.ignore_flat_ann_args(flat_ann_args, ["a"]).items()) == list(flat_ann_args.items())[1:]
        assert (
            list(parsing.ignore_flat_ann_args(flat_ann_args, [parsing.Regex("a")]).items())
            == list(flat_ann_args.items())[2:]
        )

    def test_hash_args(self):
        def f(a, *args, b=2, **kwargs):
            pass

        with pytest.raises(Exception):
            parsing.hash_args(f, (0, 1), dict(c=np.array([1, 2, 3])))
        parsing.hash_args(f, (0, 1), dict(c=np.array([1, 2, 3])), ignore_args=["c"])

    def test_get_context_vars(self):
        a = 1
        b = 2
        assert parsing.get_context_vars(["a", "b"]) == [1, 2]
        with pytest.raises(Exception):
            parsing.get_context_vars(["a", "b", "c"])
        assert parsing.get_context_vars(["c", "d", "e"], local_dict=dict(c=1, d=2, e=3)) == [1, 2, 3]
        assert parsing.get_context_vars(["c", "d", "e"], global_dict=dict(c=1, d=2, e=3)) == [1, 2, 3]


# ############# execution ############# #


def execute_func(a, *args, b=None, **kwargs):
    return a + sum(args) + b + sum(kwargs.values())


class TestExecution:
    def test_get_ray_refs(self):
        if ray_available:

            def f1(*args, **kwargs):
                pass

            def f2(*args, **kwargs):
                pass

            lst1 = [1, 2, 3]
            lst2 = [1, 2, 3]
            arr1 = np.array([1, 2, 3])
            arr2 = np.array([1, 2, 3])
            funcs_args = [
                (f1, (1, lst1, arr1), dict(a=1, b=lst1, c=arr1)),
                (f1, (2, lst2, arr2), dict(a=2, b=lst2, c=arr2)),
                (f2, (1, lst1, arr1), dict(a=1, b=lst1, c=arr1)),
            ]

            funcs_args_refs = execution.RayEngine.get_ray_refs(funcs_args, reuse_refs=False)
            func_refs = list(zip(*funcs_args_refs))[0]
            assert func_refs[0] is not func_refs[1]
            assert func_refs[0] is not func_refs[2]
            args_refs = list(zip(*funcs_args_refs))[1]
            assert args_refs[0][0] is not args_refs[1][0]
            assert args_refs[0][0] is not args_refs[2][0]
            assert args_refs[0][1] is not args_refs[1][1]
            assert args_refs[0][1] is not args_refs[2][1]
            assert args_refs[0][2] is not args_refs[1][2]
            assert args_refs[0][2] is not args_refs[2][2]
            kwargs_refs = list(zip(*funcs_args_refs))[2]
            assert kwargs_refs[0]["a"] is not kwargs_refs[1]["a"]
            assert kwargs_refs[0]["a"] is not kwargs_refs[2]["a"]
            assert kwargs_refs[0]["b"] is not kwargs_refs[1]["b"]
            assert kwargs_refs[0]["b"] is not kwargs_refs[2]["b"]
            assert kwargs_refs[0]["c"] is not kwargs_refs[1]["c"]
            assert kwargs_refs[0]["c"] is not kwargs_refs[2]["c"]

            funcs_args_refs = execution.RayEngine.get_ray_refs(funcs_args, reuse_refs=True)
            func_refs = list(zip(*funcs_args_refs))[0]
            assert func_refs[0] is func_refs[1]
            assert func_refs[0] is not func_refs[2]
            args_refs = list(zip(*funcs_args_refs))[1]
            assert args_refs[0][0] is not args_refs[1][0]
            assert args_refs[0][0] is args_refs[2][0]
            assert args_refs[0][1] is not args_refs[1][1]
            assert args_refs[0][1] is args_refs[2][1]
            assert args_refs[0][2] is not args_refs[1][2]
            assert args_refs[0][2] is args_refs[2][2]
            kwargs_refs = list(zip(*funcs_args_refs))[2]
            assert kwargs_refs[0]["a"] is not kwargs_refs[1]["a"]
            assert kwargs_refs[0]["a"] is kwargs_refs[2]["a"]
            assert kwargs_refs[0]["b"] is not kwargs_refs[1]["b"]
            assert kwargs_refs[0]["b"] is kwargs_refs[2]["b"]
            assert kwargs_refs[0]["c"] is not kwargs_refs[1]["c"]
            assert kwargs_refs[0]["c"] is kwargs_refs[2]["c"]

            assert funcs_args_refs == execution.RayEngine.get_ray_refs(funcs_args_refs)

    def test_execute(self):
        funcs_args = [
            (execute_func, (0, 1, 2), dict(b=3, c=4)),
            (execute_func, (5, 6, 7), dict(b=8, c=9)),
            (execute_func, (10, 11, 12), dict(b=13, c=14)),
        ]
        assert execution.execute(funcs_args, show_progress=True) == [10, 35, 60]
        assert execution.execute(funcs_args, engine="serial", show_progress=True) == [10, 35, 60]
        assert execution.execute(funcs_args, engine=execution.SerialEngine, show_progress=True) == [10, 35, 60]
        assert execution.execute(funcs_args, engine=execution.SerialEngine(show_progress=True)) == [10, 35, 60]
        assert execution.execute(
            funcs_args,
            engine=lambda funcs_args, my_arg: [func(*args, **kwargs) * my_arg for func, args, kwargs in funcs_args],
            my_arg=100,
        ) == [1000, 3500, 6000]
        with pytest.raises(Exception):
            execution.execute(funcs_args, engine=object)
        if dask_available:
            assert execution.execute(funcs_args, engine="dask") == [10, 35, 60]
        if ray_available:
            assert execution.execute(funcs_args, engine="ray") == [10, 35, 60]
        assert execution.execute(funcs_args, engine="threadpool") == [10, 35, 60]
        assert execution.execute(funcs_args, engine="processpool") == [10, 35, 60]
        if pathos_available:
            assert execution.execute(funcs_args, engine="pathos", pool_type="thread") == [10, 35, 60]
            assert execution.execute(funcs_args, engine="pathos", pool_type="process") == [10, 35, 60]
            assert execution.execute(funcs_args, engine="pathos", pool_type="parallel") == [10, 35, 60]

    def test_execute_chunks(self):
        def f(a, *args, b=None, **kwargs):
            return a + sum(args) + b + sum(kwargs.values())

        funcs_args = [
            (f, (0, 1, 2), dict(b=3, c=4)),
            (f, (5, 6, 7), dict(b=8, c=9)),
            (f, (10, 11, 12), dict(b=13, c=14)),
        ]

        assert execution.execute(
            funcs_args,
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
            ],
        ) == [10, 35, 60]
        assert execution.execute(
            funcs_args,
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            ],
        ) == [10, 35, 60]
        assert execution.execute(
            funcs_args,
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            ],
            in_chunk_order=True,
        ) == [35, 60, 10]
        assert execution.execute(
            funcs_args,
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=None, end=None, indices=[0]),
                chunking.ChunkMeta(uuid="", idx=0, start=None, end=None, indices=[1, 2]),
            ],
        ) == [10, 35, 60]
        assert execution.execute(
            funcs_args,
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=None, end=None, indices=[2, 1]),
                chunking.ChunkMeta(uuid="", idx=0, start=None, end=None, indices=[0]),
            ],
            in_chunk_order=True,
        ) == [60, 35, 10]

        assert execution.execute(
            iter(funcs_args),
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
            ],
        ) == [10, 35, 60]
        assert execution.execute(
            iter(funcs_args),
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            ],
        ) == [10, 35, 60]
        assert execution.execute(
            iter(funcs_args),
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            ],
            in_chunk_order=True,
        ) == [35, 60, 10]
        assert execution.execute(
            iter(funcs_args),
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=None, end=None, indices=[0]),
                chunking.ChunkMeta(uuid="", idx=0, start=None, end=None, indices=[1, 2]),
            ],
        ) == [10, 35, 60]
        assert execution.execute(
            iter(funcs_args),
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=None, end=None, indices=[2, 1]),
                chunking.ChunkMeta(uuid="", idx=0, start=None, end=None, indices=[0]),
            ],
            in_chunk_order=True,
        ) == [60, 35, 10]

        assert execution.execute(iter(funcs_args), n_chunks=1) == [10, 35, 60]
        assert execution.execute(iter(funcs_args), n_chunks=2) == [10, 35, 60]
        assert execution.execute(iter(funcs_args), n_chunks=3) == [10, 35, 60]
        assert execution.execute(iter(funcs_args), chunk_len=1) == [10, 35, 60]
        assert execution.execute(iter(funcs_args), chunk_len=2) == [10, 35, 60]
        assert execution.execute(iter(funcs_args), chunk_len=3) == [10, 35, 60]
        assert execution.execute(iter(funcs_args), chunk_len="auto") == [10, 35, 60]


# ############# pickling ############# #


class TestPickling:
    def test_pdict(self, tmp_path):
        index = pd.date_range("2023", periods=5)
        columns = pd.Index(["a", "b", "c"], name="symbol")
        wrapper = vbt.ArrayWrapper(index, columns)
        acc1 = vbt.GenericAccessor(wrapper, wrapper.fill(0).values)
        acc2 = vbt.GenericAccessor(wrapper, wrapper.fill(1).values)
        d1 = dict(acc1=acc1)
        d2 = dict(acc1=acc1, acc2=acc2)
        d3 = d2
        d4 = dict(a=dict(b=dict(d3=d3)))
        pdict = pickling.pdict(hello="world", cls=vbt.ArrayWrapper, d1=d1, d2=d2, d3=d3, d4=d4)
        pdict.save(tmp_path / "pdict")
        assert pickling.pdict.load(tmp_path / "pdict") == pdict
        pdict.save(tmp_path / "pdict", rec_state_only=True)
        assert pickling.pdict.load(tmp_path / "pdict") == pdict
        pdict.save(tmp_path / "pdict", file_format="ini")
        assert pickling.pdict.load(tmp_path / "pdict", file_format="ini") == pdict
        pdict.save(tmp_path / "pdict", file_format="ini", nested=False)
        assert pickling.pdict.load(tmp_path / "pdict", file_format="ini") == pdict
        pdict.save(tmp_path / "pdict", file_format="ini", use_refs=False)
        assert pickling.pdict.load(tmp_path / "pdict", file_format="ini", use_refs=False) == pdict
        pdict.save(tmp_path / "pdict", file_format="ini", use_class_ids=False)
        assert pickling.pdict.load(tmp_path / "pdict", file_format="ini", use_class_ids=False) == pdict

    def test_compression(self, tmp_path):
        vbt.Config(a=0).save(tmp_path)
        with pytest.raises(Exception):
            vbt.Config(a=1).save(tmp_path, compression=True)
        vbt.Config(a=2).save(tmp_path, compression=False)
        vbt.Config(a=3).save(tmp_path, compression="gzip")
        vbt.Config(a=4).save(tmp_path, compression="gz")
        vbt.Config(a=5).save(tmp_path, compression="bz2")
        assert vbt.Config.load(tmp_path)["a"] == 2
        assert vbt.Config.load(tmp_path, compression=False)["a"] == 2
        assert vbt.Config.load(tmp_path, compression="gzip")["a"] == 3
        assert vbt.Config.load(tmp_path, compression="gz")["a"] == 4
        assert vbt.Config.load(tmp_path, compression="bz")["a"] == 5
        (tmp_path / "Config.pickle").unlink()
        with pytest.raises(Exception):
            vbt.Config.load(tmp_path)
        (tmp_path / "Config.pickle.bz2").unlink()
        with pytest.raises(Exception):
            vbt.Config.load(tmp_path, compression="bz")


# ############# chunking ############# #


class TestChunking:
    def test_arg_getter_mixin(self):
        def f(a, *args, b=None, **kwargs):
            pass

        ann_args = parsing.annotate_args(f, (0, 1), dict(c=2))

        assert chunking.ArgGetter(0).get_arg(ann_args) == 0
        assert chunking.ArgGetter("a").get_arg(ann_args) == 0
        assert chunking.ArgGetter(1).get_arg(ann_args) == 1
        assert chunking.ArgGetter(2).get_arg(ann_args) is None
        assert chunking.ArgGetter("b").get_arg(ann_args) is None
        assert chunking.ArgGetter(3).get_arg(ann_args) == 2
        assert chunking.ArgGetter("c").get_arg(ann_args) == 2
        with pytest.raises(Exception):
            chunking.ArgGetter(4).get_arg(ann_args)
        with pytest.raises(Exception):
            chunking.ArgGetter("d").get_arg(ann_args)

    def test_sizers(self):
        def f(a):
            pass

        ann_args = parsing.annotate_args(f, (10,), {})
        assert chunking.ArgSizer(arg_query="a").apply(ann_args) == 10

        ann_args = parsing.annotate_args(f, ([1, 2, 3],), {})
        assert chunking.LenSizer(arg_query="a").apply(ann_args) == 3

        ann_args = parsing.annotate_args(f, (3,), {})
        assert chunking.LenSizer(arg_query="a", single_type=int).apply(ann_args) == 1
        with pytest.raises(Exception):
            chunking.LenSizer().apply(ann_args)
        with pytest.raises(Exception):
            chunking.LenSizer(arg_query="a").apply(ann_args)

        ann_args = parsing.annotate_args(f, ((2, 3),), {})
        with pytest.raises(Exception):
            chunking.ShapeSizer().apply(ann_args)
        with pytest.raises(Exception):
            chunking.ShapeSizer(arg_query="a").apply(ann_args)
        assert chunking.ShapeSizer(arg_query="a", axis=0).apply(ann_args) == 2
        assert chunking.ShapeSizer(arg_query="a", axis=1).apply(ann_args) == 3
        assert chunking.ShapeSizer(arg_query="a", axis=2).apply(ann_args) == 0

        ann_args = parsing.annotate_args(f, (np.empty((2, 3)),), {})
        with pytest.raises(Exception):
            chunking.ArraySizer().apply(ann_args)
        with pytest.raises(Exception):
            chunking.ArraySizer(arg_query="a").apply(ann_args)
        assert chunking.ArraySizer(arg_query="a", axis=0).apply(ann_args) == 2
        assert chunking.ArraySizer(arg_query="a", axis=1).apply(ann_args) == 3
        assert chunking.ArraySizer(arg_query="a", axis=2).apply(ann_args) == 0

    def test_yield_chunk_meta(self):
        with pytest.raises(Exception):
            list(chunking.yield_chunk_meta(n_chunks=0))

        chunk_meta_equal(
            list(chunking.yield_chunk_meta(n_chunks=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=None, end=None, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=None, end=None, indices=None),
                chunking.ChunkMeta(uuid="", idx=2, start=None, end=None, indices=None),
                chunking.ChunkMeta(uuid="", idx=3, start=None, end=None, indices=None),
            ],
        )
        chunk_meta_equal(
            list(chunking.yield_chunk_meta(n_chunks=1, size=4)),
            [chunking.ChunkMeta(uuid="", idx=0, start=0, end=4, indices=None)],
        )
        chunk_meta_equal(
            list(chunking.yield_chunk_meta(n_chunks=2, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=2, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=2, end=4, indices=None),
            ],
        )
        chunk_meta_equal(
            list(chunking.yield_chunk_meta(n_chunks=3, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=2, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=2, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=2, start=3, end=4, indices=None),
            ],
        )
        chunk_meta_equal(
            list(chunking.yield_chunk_meta(n_chunks=4, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=2, indices=None),
                chunking.ChunkMeta(uuid="", idx=2, start=2, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=3, start=3, end=4, indices=None),
            ],
        )
        chunk_meta_equal(
            list(chunking.yield_chunk_meta(n_chunks=5, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=2, indices=None),
                chunking.ChunkMeta(uuid="", idx=2, start=2, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=3, start=3, end=4, indices=None),
            ],
        )
        with pytest.raises(Exception):
            list(chunking.yield_chunk_meta(chunk_len=0, size=4))
        chunk_meta_equal(
            list(chunking.yield_chunk_meta(chunk_len=1, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=2, indices=None),
                chunking.ChunkMeta(uuid="", idx=2, start=2, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=3, start=3, end=4, indices=None),
            ],
        )
        chunk_meta_equal(
            list(chunking.yield_chunk_meta(chunk_len=2, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=2, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=2, end=4, indices=None),
            ],
        )
        chunk_meta_equal(
            list(chunking.yield_chunk_meta(chunk_len=3, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=3, end=4, indices=None),
            ],
        )
        chunk_meta_equal(
            list(chunking.yield_chunk_meta(chunk_len=4, size=4)),
            [chunking.ChunkMeta(uuid="", idx=0, start=0, end=4, indices=None)],
        )
        chunk_meta_equal(
            list(chunking.yield_chunk_meta(chunk_len=5, size=4)),
            [chunking.ChunkMeta(uuid="", idx=0, start=0, end=4, indices=None)],
        )
        chunk_meta_equal(
            list(chunking.yield_chunk_meta(n_chunks=2, size=2, min_size=2)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=2, indices=None),
            ],
        )
        chunk_meta_equal(
            list(chunking.yield_chunk_meta(n_chunks=2, size=2, min_size=3)),
            [chunking.ChunkMeta(uuid="", idx=0, start=0, end=2, indices=None)],
        )
        with pytest.raises(Exception):
            list(chunking.yield_chunk_meta(n_chunks=2, size=4, chunk_len=2))

    def test_chunk_meta_generators(self):
        def f(a):
            pass

        chunk_meta = [
            chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
            chunking.ChunkMeta(uuid="", idx=2, start=3, end=6, indices=None),
        ]
        ann_args = parsing.annotate_args(f, (chunk_meta,), {})
        chunk_meta_equal(list(chunking.ArgChunkMeta("a").get_chunk_meta(ann_args)), chunk_meta)

        ann_args = parsing.annotate_args(f, ([1, 2, 3],), {})
        chunk_meta_equal(list(chunking.LenChunkMeta("a").get_chunk_meta(ann_args)), chunk_meta)

    def test_get_chunk_meta_from_args(self):
        def f(a, *args, b=None, **kwargs):
            pass

        chunk_meta = [
            chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            chunking.ChunkMeta(uuid="", idx=1, start=1, end=2, indices=None),
            chunking.ChunkMeta(uuid="", idx=2, start=2, end=3, indices=None),
        ]

        ann_args = parsing.annotate_args(f, (2, 3, 1), dict(b=[1, 2, 3]))
        chunk_meta_equal(list(chunking.get_chunk_meta_from_args(ann_args, size=3, n_chunks=3)), chunk_meta)
        chunk_meta_equal(
            list(
                chunking.get_chunk_meta_from_args(
                    ann_args,
                    size=3,
                    n_chunks=lambda ann_args: ann_args["args"]["value"][0],
                )
            ),
            chunk_meta,
        )
        chunk_meta_equal(
            list(chunking.get_chunk_meta_from_args(ann_args, size=3, n_chunks=chunking.ArgSizer(arg_query=1))),
            chunk_meta,
        )
        with pytest.raises(Exception):
            list(chunking.get_chunk_meta_from_args(ann_args, size=3, n_chunks="a"))

        chunk_meta_equal(list(chunking.get_chunk_meta_from_args(ann_args, chunk_len=1, size=3)), chunk_meta)
        chunk_meta_equal(
            list(
                chunking.get_chunk_meta_from_args(
                    ann_args,
                    chunk_len=1,
                    size=lambda ann_args: ann_args["args"]["value"][0],
                )
            ),
            chunk_meta,
        )
        chunk_meta_equal(
            list(chunking.get_chunk_meta_from_args(ann_args, chunk_len=1, size=chunking.ArgSizer(arg_query=1))),
            chunk_meta,
        )
        with pytest.raises(Exception):
            list(chunking.get_chunk_meta_from_args(ann_args, chunk_len=1, size="a"))

        chunk_meta_equal(list(chunking.get_chunk_meta_from_args(ann_args, size=3, chunk_len=1)), chunk_meta)
        chunk_meta_equal(
            list(
                chunking.get_chunk_meta_from_args(
                    ann_args,
                    size=3,
                    chunk_len=lambda ann_args: ann_args["args"]["value"][1],
                )
            ),
            chunk_meta,
        )
        chunk_meta_equal(
            list(chunking.get_chunk_meta_from_args(ann_args, size=3, chunk_len=chunking.ArgSizer(arg_query=2))),
            chunk_meta,
        )
        with pytest.raises(Exception):
            list(chunking.get_chunk_meta_from_args(ann_args, size=3, chunk_len="a"))

        chunk_meta_equal(list(chunking.get_chunk_meta_from_args(ann_args, chunk_meta=chunk_meta)), chunk_meta)
        chunk_meta_equal(
            list(chunking.get_chunk_meta_from_args(ann_args, chunk_meta=chunking.LenChunkMeta("b"))),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=2, start=3, end=6, indices=None),
            ],
        )
        chunk_meta_equal(
            list(chunking.get_chunk_meta_from_args(ann_args, chunk_meta=lambda ann_args: chunk_meta)),
            chunk_meta,
        )

    def test_take_from_args(self):
        def f(a, b, *args, c=None, d=None, **kwargs):
            pass

        lst = [0, 1, 2]

        ann_args = parsing.annotate_args(
            f,
            (lst, lst, lst, (lst, lst)),
            dict(c=lst, d=lst, e=lst, f=dict(g=lst, h=lst)),
        )
        arg_take_spec = dict(
            b=chunking.ChunkSelector(),
            args=chunking.ArgsTaker(None, chunking.SequenceTaker(cont_take_spec=(None, chunking.ChunkSlicer()))),
            d=chunking.ChunkSelector(),
            kwargs=chunking.KwargsTaker(f=chunking.MappingTaker(cont_take_spec=dict(h=chunking.ChunkSlicer()))),
        )
        args, kwargs = chunking.take_from_args(
            ann_args,
            arg_take_spec,
            chunking.ChunkMeta(uuid="", idx=0, start=1, end=3, indices=None),
        )
        assert args == (lst, lst[0], lst, (lst, lst[1:3]))
        assert kwargs == dict(c=lst, d=lst[0], e=lst, f=dict(g=lst, h=lst[1:3]))

    def test_chunk_takers(self):
        a = np.arange(6).reshape((2, 3))
        sr = pd.Series(a[:, 0])
        df = pd.DataFrame(a)

        assert chunking.ChunkSelector().apply([1, 2, 3], chunking.ChunkMeta("", 0, 0, 1, None)) == 1
        assert chunking.ChunkSelector(keep_dims=True).apply([1, 2, 3], chunking.ChunkMeta("", 0, 0, 1, None)) == [1]
        assert chunking.ChunkSelector().apply(None, chunking.ChunkMeta("", 0, 0, 1, None)) is None
        with pytest.raises(Exception):
            chunking.ChunkSelector(ignore_none=False).apply(None, chunking.ChunkMeta("", 0, 0, 1, None))
        assert chunking.ChunkSelector(single_type=int).apply(10, chunking.ChunkMeta("", 0, 0, 1, None)) == 10
        with pytest.raises(Exception):
            chunking.ChunkSelector().apply(10, chunking.ChunkMeta("", 0, 0, 1, None))
        assert chunking.ChunkSlicer().apply([1, 2, 3], chunking.ChunkMeta("", 0, 0, 1, None)) == [1]
        np.testing.assert_array_equal(
            chunking.ChunkSlicer().apply(np.array([1, 2, 3]), chunking.ChunkMeta("", 0, None, None, np.array([0, 0]))),
            np.array([1, 1]),
        )
        with pytest.raises(Exception):
            chunking.ChunkSlicer().apply(np.array([1, 2, 3]), chunking.ChunkMeta("", 0, None, None, np.array([3])))

        assert chunking.CountAdapter().apply(10, chunking.ChunkMeta("", 0, 0, 1, None)) == 1
        assert chunking.CountAdapter().apply(10, chunking.ChunkMeta("", 0, 8, 12, None)) == 2
        assert chunking.CountAdapter().apply(10, chunking.ChunkMeta("", 0, 12, 13, None)) == 0

        assert chunking.ShapeSelector(axis=0).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None)) == (2, 3)
        assert chunking.ShapeSelector(axis=1).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None)) == (1, 3)
        assert chunking.ShapeSelector(axis=2).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None)) == (1, 2)
        with pytest.raises(Exception):
            chunking.ShapeSelector(axis=4).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None))
        assert chunking.ShapeSelector(axis=0, keep_dims=True).apply(
            (1, 2, 3),
            chunking.ChunkMeta("", 0, 0, 1, None),
        ) == (1, 2, 3)
        with pytest.raises(Exception):
            chunking.ShapeSelector(axis=0).apply((1, 2, 3), chunking.ChunkMeta("", 1, 0, 1, None))
        assert chunking.ShapeSelector(axis=0).apply((1,), chunking.ChunkMeta("", 0, 0, 1, None)) == ()
        assert chunking.ShapeSlicer(axis=0).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None)) == (1, 2, 3)
        assert chunking.ShapeSlicer(axis=1).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None)) == (1, 1, 3)
        assert chunking.ShapeSlicer(axis=2).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None)) == (1, 2, 1)
        with pytest.raises(Exception):
            chunking.ShapeSlicer(axis=4).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None))
        assert chunking.ShapeSlicer(axis=0).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 2, None)) == (1, 2, 3)
        assert chunking.ShapeSlicer(axis=0).apply((1, 2, 3), chunking.ChunkMeta("", 0, 1, 2, None)) == (2, 3)
        assert chunking.ShapeSlicer(axis=0).apply(
            (1, 2, 3),
            chunking.ChunkMeta("", 0, None, None, np.array([0, 0])),
        ) == (2, 2, 3)
        with pytest.raises(Exception):
            chunking.ShapeSlicer(axis=0).apply((1, 2, 3), chunking.ChunkMeta("", 0, None, None, np.array([1])))

        np.testing.assert_array_equal(
            chunking.ArraySelector(axis=0).apply(a, chunking.ChunkMeta("", 0, 0, 1, None)),
            a[0],
        )
        np.testing.assert_array_equal(
            chunking.ArraySelector(axis=0, keep_dims=True).apply(a, chunking.ChunkMeta("", 0, 0, 1, None)),
            a[[0]],
        )
        np.testing.assert_array_equal(
            chunking.ArraySelector(axis=1).apply(a, chunking.ChunkMeta("", 0, 0, 1, None)),
            a[:, 0],
        )
        with pytest.raises(Exception):
            chunking.ArraySelector(axis=2).apply(a, chunking.ChunkMeta("", 0, 0, 1, None))
        assert chunking.ArraySelector(axis=0).apply(sr, chunking.ChunkMeta("", 0, 0, 1, None)) == sr.iloc[0]
        assert_series_equal(
            chunking.ArraySelector(axis=1).apply(df, chunking.ChunkMeta("", 0, 0, 1, None)),
            df.iloc[:, 0],
        )
        np.testing.assert_array_equal(
            chunking.ArraySlicer(axis=0).apply(a, chunking.ChunkMeta("", 0, 0, 1, None)),
            a[[0]],
        )
        np.testing.assert_array_equal(
            chunking.ArraySlicer(axis=1).apply(a, chunking.ChunkMeta("", 0, 0, 1, None)),
            a[:, [0]],
        )
        np.testing.assert_array_equal(
            chunking.ArraySlicer(axis=0).apply(a, chunking.ChunkMeta("", 0, None, None, np.array([0]))),
            a[[0]],
        )
        with pytest.raises(Exception):
            chunking.ArraySlicer(axis=0).apply(a, chunking.ChunkMeta("", 0, None, None, np.array([2])))
        with pytest.raises(Exception):
            chunking.ArraySlicer(axis=2).apply(a, chunking.ChunkMeta("", 0, 0, 1, None))
        assert_series_equal(
            chunking.ArraySlicer(axis=0).apply(sr, chunking.ChunkMeta("", 0, 0, 1, None)),
            sr.iloc[[0]],
        )
        assert_frame_equal(
            chunking.ArraySlicer(axis=1).apply(df, chunking.ChunkMeta("", 0, 0, 1, None)),
            df.iloc[:, [0]],
        )

    def test_yield_arg_chunks(self):
        def f(a, *args, b=None, **kwargs):
            pass

        ann_args = parsing.annotate_args(f, (2, 3, 1), dict(b=[1, 2, 3]))
        chunk_meta = [
            chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            chunking.ChunkMeta(uuid="", idx=1, start=1, end=2, indices=None),
            chunking.ChunkMeta(uuid="", idx=2, start=2, end=3, indices=None),
        ]
        arg_take_spec = dict(b=chunking.ChunkSelector())
        result = [(f, (2, 3, 1), {"b": 1}), (f, (2, 3, 1), {"b": 2}), (f, (2, 3, 1), {"b": 3})]
        assert list(chunking.yield_arg_chunks(f, ann_args, chunk_meta, arg_take_spec=arg_take_spec)) == result
        assert (
            list(
                chunking.yield_arg_chunks(
                    f,
                    ann_args,
                    chunk_meta,
                    arg_take_spec=lambda ann_args, chunk_meta: ((2, 3, 1), dict(b=[1, 2, 3][chunk_meta.idx])),
                )
            )
            == result
        )
        ann_args = parsing.annotate_args(
            f,
            (template.RepEval('ann_args["args"]["value"][1] + 1'), 3, 1),
            dict(b=template.Rep("lst")),
        )
        assert (
            list(
                chunking.yield_arg_chunks(
                    f,
                    ann_args,
                    chunk_meta,
                    arg_take_spec=arg_take_spec,
                    template_context={"lst": [1, 2, 3]},
                )
            )
            == result
        )

    def test_chunked(self):
        @chunking.chunked(n_chunks=2, size=vbt.LenSizer(arg_query="a"), arg_take_spec=dict(a=vbt.ChunkSlicer()))
        def f(a):
            return a

        results = f(np.arange(10))
        np.testing.assert_array_equal(results[0], np.arange(5))
        np.testing.assert_array_equal(results[1], np.arange(5, 10))

        f.options.n_chunks = 3
        results = f(np.arange(10))
        np.testing.assert_array_equal(results[0], np.arange(4))
        np.testing.assert_array_equal(results[1], np.arange(4, 7))
        np.testing.assert_array_equal(results[2], np.arange(7, 10))

        results = f(np.arange(10), _n_chunks=4)
        np.testing.assert_array_equal(results[0], np.arange(3))
        np.testing.assert_array_equal(results[1], np.arange(3, 6))
        np.testing.assert_array_equal(results[2], np.arange(6, 8))
        np.testing.assert_array_equal(results[3], np.arange(8, 10))

        results = f(np.arange(10), _n_chunks=1, _skip_one_chunk=False)
        np.testing.assert_array_equal(results[0], np.arange(10))

        results = f(np.arange(10), _n_chunks=1, _skip_one_chunk=True)
        np.testing.assert_array_equal(results, np.arange(10))

        @vbt.chunked(n_chunks=2, size=vbt.LenSizer(arg_query="a"))
        def f2(chunk_meta, a):
            return a[chunk_meta.start : chunk_meta.end]

        results = f2(np.arange(10))
        np.testing.assert_array_equal(results[0], np.arange(5))
        np.testing.assert_array_equal(results[1], np.arange(5, 10))

        @vbt.chunked(n_chunks=2, size=vbt.LenSizer(arg_query="a"), prepend_chunk_meta=False)
        def f3(chunk_meta, a):
            return a[chunk_meta.start : chunk_meta.end]

        results = f3(template.Rep("chunk_meta"), np.arange(10))
        np.testing.assert_array_equal(results[0], np.arange(5))
        np.testing.assert_array_equal(results[1], np.arange(5, 10))

        with pytest.raises(Exception):

            @vbt.chunked(n_chunks=2, size=vbt.LenSizer(arg_query="a"), prepend_chunk_meta=True)
            def f4(chunk_meta, a):
                return a[chunk_meta.start : chunk_meta.end]

            f4(template.Rep("chunk_meta"), np.arange(10))

        @vbt.chunked(
            n_chunks=2,
            size=lambda ann_args: len(ann_args["a"]["value"]),
            arg_take_spec=dict(a=vbt.ChunkSlicer()),
        )
        def f5(a):
            return a

        results = f5(np.arange(10))
        np.testing.assert_array_equal(results[0], np.arange(5))
        np.testing.assert_array_equal(results[1], np.arange(5, 10))

        def arg_take_spec(ann_args, chunk_meta, **kwargs):
            a = ann_args["a"]["value"]
            lens = ann_args["lens"]["value"]
            lens_chunk = lens[chunk_meta.start : chunk_meta.end]
            a_end = np.cumsum(lens)
            a_start = a_end - lens
            a_start = a_start[chunk_meta.start : chunk_meta.end][0]
            a_end = a_end[chunk_meta.start : chunk_meta.end][-1]
            a_chunk = a[a_start:a_end]
            return (a_chunk, lens_chunk), {}

        @vbt.chunked(
            n_chunks=2,
            size=vbt.LenSizer(arg_query="lens"),
            arg_take_spec=arg_take_spec,
            merge_func=lambda results: [list(r) for r in results],
        )
        def f6(a, lens):
            ends = np.cumsum(lens)
            starts = ends - lens
            for i in range(len(lens)):
                yield a[starts[i] : ends[i]]

        results = f6(np.arange(10), [1, 2, 3, 4])
        np.testing.assert_array_equal(results[0][0], np.arange(1))
        np.testing.assert_array_equal(results[0][1], np.arange(1, 3))
        np.testing.assert_array_equal(results[1][0], np.arange(3, 6))
        np.testing.assert_array_equal(results[1][1], np.arange(6, 10))

        if dask_available:

            @vbt.chunked(
                n_chunks=2,
                size=vbt.LenSizer(arg_query="a"),
                arg_take_spec=dict(a=vbt.ChunkSlicer()),
                merge_func=np.concatenate,
                engine="dask",
            )
            def f7(a):
                return a

            np.testing.assert_array_equal(f7(np.arange(10)), np.arange(10))

        if ray_available:

            @vbt.chunked(
                n_chunks=2,
                size=vbt.LenSizer(arg_query="a"),
                arg_take_spec=dict(a=vbt.ChunkSlicer()),
                merge_func=np.concatenate,
                engine="ray",
            )
            def f8(a):
                return a

            np.testing.assert_array_equal(f8(np.arange(10)), np.arange(10))


# ############# jitting ############# #


class TestJitting:
    def test_jitters(self):
        py_func = lambda x: x

        assert jitting.NumPyJitter().decorate(py_func) is py_func
        if checks.is_numba_enabled():
            assert isinstance(jitting.NumbaJitter().decorate(py_func), CPUDispatcher)
            assert not jitting.NumbaJitter(parallel=True).decorate(py_func).targetoptions["parallel"]
            assert jitting.NumbaJitter(parallel=True).decorate(py_func, tags={"can_parallel"}).targetoptions["parallel"]
            assert (
                jitting.NumbaJitter(parallel=True, fix_cannot_parallel=False)
                .decorate(py_func)
                .targetoptions["parallel"]
            )

    def test_get_func_suffix(self):
        def py_func():
            pass

        def func_nb():
            pass

        assert jitting.get_func_suffix(lambda x: x) is None
        assert jitting.get_func_suffix(py_func) is None
        assert jitting.get_func_suffix(func_nb) == "nb"

    def test_resolve_jitter_type(self):
        def py_func():
            pass

        def func_nb():
            pass

        with pytest.raises(Exception):
            jitting.resolve_jitter_type()
        with pytest.raises(Exception):
            jitting.resolve_jitter_type(py_func=py_func)
        assert jitting.resolve_jitter_type(py_func=func_nb) is jitting.NumbaJitter
        assert jitting.resolve_jitter_type(jitter="numba", py_func=func_nb) is jitting.NumbaJitter
        with pytest.raises(Exception):
            jitting.resolve_jitter_type(jitter="numba2", py_func=func_nb)
        assert jitting.resolve_jitter_type(jitter=jitting.NumbaJitter, py_func=func_nb) is jitting.NumbaJitter
        assert jitting.resolve_jitter_type(jitter=jitting.NumbaJitter(), py_func=func_nb) is jitting.NumbaJitter
        with pytest.raises(Exception):
            jitting.resolve_jitter_type(jitter=object, py_func=func_nb)

    def test_get_id_of_jitter_type(self):
        assert jitting.get_id_of_jitter_type(jitting.NumbaJitter) == "nb"
        assert jitting.get_id_of_jitter_type(jitting.NumPyJitter) == "np"
        assert jitting.get_id_of_jitter_type(object) is None

    def test_resolve_jitted_kwargs(self):
        assert jitting.resolve_jitted_kwargs(option=True) == dict()
        assert jitting.resolve_jitted_kwargs(option=False) is None
        assert jitting.resolve_jitted_kwargs(option=dict(test="test")) == dict(test="test")
        assert jitting.resolve_jitted_kwargs(option="numba") == dict(jitter="numba")
        with pytest.raises(Exception):
            jitting.resolve_jitted_kwargs(option=10)
        assert jitting.resolve_jitted_kwargs(option="numba", jitter="numpy") == dict(jitter="numba")

    def test_jitted(self):
        class MyJitter(jitting.Jitter):
            def decorate(self, py_func, tags=None):
                @wraps(py_func)
                def wrapper(*args, **kwargs):
                    return py_func(*args, **kwargs)

                wrapper.config = self.config
                return wrapper

        vbt.settings.jitting.jitters["my"] = dict(cls=MyJitter)

        @jitting.jitted
        def func_my():
            pass

        assert dict(func_my.config) == dict()

        @jitting.jitted(test="test")
        def func_my():
            pass

        assert dict(func_my.config) == dict(test="test")
