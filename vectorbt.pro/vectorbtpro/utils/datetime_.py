# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for working with dates and time."""

import attr
import warnings
from datetime import datetime, timezone, timedelta, tzinfo, date, time
from collections import namedtuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import re

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "DTC",
]

PandasDatetimeIndex = (pd.DatetimeIndex, pd.PeriodIndex)


def split_freq_str(freq: str) -> tp.Optional[tp.Tuple[int, str]]:
    """Split (human-readable) frequency into multiplier and unambiguous unit.

    Can be used both as offset and timedelta.

    The following units are returned:
    * "s" for second
    * "t" for minute
    * "h" for hour
    * "d" for day
    * "W" for week
    * "M" for month
    * "Q" for quarter
    * "Y" for year"""

    freq = "".join(freq.strip().split())
    match = re.match(r"^(\d*)\s*([a-zA-Z-]+)$", freq)
    if match.group(1) == "" and match.group(2).isnumeric():
        raise ValueError("Frequency must contain unit")
    if not match:
        return None
    if match.group(1) == "":
        multiplier = 1
    else:
        multiplier = int(match.group(1))
    if match.group(2) == "":
        raise ValueError("Frequency must contain unit")
    else:
        unit = match.group(2)
    if unit in ("S", "sec", "second", "seconds"):
        unit = "s"
    elif unit in ("T", "m", "min", "minute", "minutes"):
        unit = "t"
    elif unit in ("H", "hour", "hours", "hourly"):
        unit = "h"
    elif unit in ("D", "day", "days", "daily"):
        unit = "d"
    elif unit in ("w", "wk", "week", "weeks", "weekly"):
        unit = "W"
    elif unit in ("mo", "month", "months", "monthly"):
        unit = "M"
    elif unit in ("q", "quarter", "quarters", "quarterly"):
        unit = "Q"
    elif unit in ("y", "year", "years", "yearly", "annual", "annually"):
        unit = "Y"
    return multiplier, unit


def prepare_freq(freq: tp.FrequencyLike) -> tp.FrequencyLike:
    """Prepare frequency using `split_freq_str`.

    To include multiple units, separate them with comma."""
    if isinstance(freq, str):
        if "," in freq:
            new_freq = ""
            for _freq in freq.split(","):
                split = split_freq_str(_freq)
                if split is not None:
                    new_freq += str(split[0]) + str(split[1])
                else:
                    return freq
            return new_freq
        split = split_freq_str(freq)
        if split is not None:
            freq = str(split[0]) + str(split[1])
        return freq
    return freq


def freq_to_timedelta(freq: tp.FrequencyLike) -> pd.Timedelta:
    """Convert a frequency-like object to `pd.Timedelta`."""
    if not isinstance(freq, pd.Timedelta):
        if isinstance(freq, str) and freq.startswith("-"):
            neg_td = True
            freq = freq[1:]
        else:
            neg_td = False
        freq = prepare_freq(freq)
        if isinstance(freq, str) and not freq[0].isdigit():
            # Otherwise "ValueError: unit abbreviation w/o a number"
            freq = pd.Timedelta(1, unit=freq)
        else:
            freq = pd.Timedelta(freq)
        if neg_td:
            freq = -freq
    if hasattr(freq, "unit") and freq.unit != "ns":
        freq = freq.as_unit("ns", round_ok=False)
    return freq


def parse_timedelta(td: tp.TimedeltaLike) -> tp.Union[pd.Timedelta, pd.DateOffset]:
    """Parse a timedelta-like object into Pandas format."""
    if isinstance(td, (pd.Timedelta, pd.DateOffset)):
        return td
    try:
        return to_offset(prepare_freq(td))
    except Exception as e:
        return freq_to_timedelta(td)


DTCNT = namedtuple("DTCNT", ["year", "month", "day", "weekday", "hour", "minute", "second", "nanosecond"])
"""Named tuple version of `DTC`."""


DTCT = tp.TypeVar("DTCT", bound="DTC")


@attr.s(frozen=True)
class DTC:
    """Class representing one or more datetime components."""

    year: tp.Optional[int] = attr.ib(default=None)
    """Year."""

    month: tp.Optional[int] = attr.ib(default=None)
    """Month."""

    day: tp.Optional[int] = attr.ib(default=None)
    """Day of month."""

    weekday: tp.Optional[int] = attr.ib(default=None)
    """Day of week."""

    hour: tp.Optional[int] = attr.ib(default=None)
    """Hour."""

    minute: tp.Optional[int] = attr.ib(default=None)
    """Minute."""

    second: tp.Optional[int] = attr.ib(default=None)
    """Second."""

    nanosecond: tp.Optional[int] = attr.ib(default=None)
    """Nanosecond."""

    @classmethod
    def from_datetime(cls: tp.Type[DTCT], dt: tp.Datetime) -> DTCT:
        """Get `DTC` instance from a `datetime.datetime` object."""
        if isinstance(dt, np.datetime64):
            dt = pd.Timestamp(dt)
        if isinstance(dt, pd.Timestamp):
            nanosecond = dt.microsecond * 1000 + dt.nanosecond
            dt = dt.to_pydatetime(warn=False)
        else:
            nanosecond = dt.microsecond * 1000
        return cls(
            year=dt.year,
            month=dt.month,
            day=dt.day,
            weekday=dt.weekday(),
            hour=dt.hour,
            minute=dt.minute,
            second=dt.second,
            nanosecond=nanosecond,
        )

    @classmethod
    def from_date(cls: tp.Type[DTCT], d: date) -> DTCT:
        """Get `DTC` instance from a `datetime.date` object."""
        return cls(year=d.year, month=d.month, day=d.day, weekday=d.weekday())

    @classmethod
    def from_time(cls: tp.Type[DTCT], t: time) -> DTCT:
        """Get `DTC` instance from a `datetime.time` object."""
        return cls(hour=t.hour, minute=t.minute, second=t.second, nanosecond=t.microsecond * 1000)

    @classmethod
    def parse_time_str(cls: tp.Type[DTCT], time_str: str, **parse_kwargs) -> DTCT:
        """Parse `DTC` instance from a time string."""
        from dateutil.parser import parser

        result = parser()._parse(time_str, **parse_kwargs)[0]
        if result.microsecond is None:
            nanosecond = None
        else:
            nanosecond = result.microsecond * 1000
        return cls(
            year=result.year,
            month=result.month,
            day=result.day,
            weekday=result.weekday,
            hour=result.hour,
            minute=result.minute,
            second=result.second,
            nanosecond=nanosecond,
        )

    @classmethod
    def from_namedtuple(cls: tp.Type[DTCT], dtc: DTCNT) -> DTCT:
        """Get `DTC` instance from a named tuple of the type `DTCNT`."""
        return cls(
            year=dtc.year,
            month=dtc.month,
            day=dtc.day,
            weekday=dtc.weekday,
            hour=dtc.hour,
            minute=dtc.minute,
            second=dtc.second,
            nanosecond=dtc.nanosecond,
        )

    @classmethod
    def parse(cls: tp.Type[DTCT], dtc_like: tp.DTCLike, **parse_kwargs) -> DTCT:
        """Parse `DTC` instance from a datetime-component-like object."""
        if checks.is_namedtuple(dtc_like):
            return cls.from_namedtuple(dtc_like)
        if isinstance(dtc_like, np.datetime64):
            dtc_like = pd.Timestamp(dtc_like)
        if isinstance(dtc_like, pd.Timestamp):
            if dtc_like.tzinfo is not None:
                raise ValueError("DTC doesn't support timezones")
            dtc_like = dtc_like.to_pydatetime()
        if isinstance(dtc_like, datetime):
            if dtc_like.tzinfo is not None:
                raise ValueError("DTC doesn't support timezones")
            return cls.from_datetime(dtc_like)
        if isinstance(dtc_like, date):
            return cls.from_date(dtc_like)
        if isinstance(dtc_like, time):
            return cls.from_time(dtc_like)
        if isinstance(dtc_like, (int, str)):
            return cls.parse_time_str(str(dtc_like), **parse_kwargs)
        raise TypeError(f"Invalid type {type(dtc_like)}")

    @classmethod
    def is_parsable(
        cls: tp.Type[DTCT],
        dtc_like: tp.DTCLike,
        check_func: tp.Optional[tp.Callable] = None,
        **parse_kwargs,
    ) -> bool:
        """Check whether a datetime-component-like object is parsable."""
        try:
            if isinstance(dtc_like, DTC):
                return True
            dtc = cls.parse(dtc_like, **parse_kwargs)
            if check_func is not None and not check_func(dtc):
                return False
            return True
        except Exception as e:
            pass
        return False

    def has_date(self) -> bool:
        """Whether any date component is set."""
        return self.year is not None or self.month is not None or self.day is not None

    def has_full_date(self) -> bool:
        """Whether all date components are set."""
        return self.year is not None and self.month is not None and self.day is not None

    def has_weekday(self) -> bool:
        """Whether the weekday component is set."""
        return self.weekday is not None

    def has_time(self) -> bool:
        """Whether any time component is set."""
        return (
            self.hour is not None or self.minute is not None or self.second is not None or self.nanosecond is not None
        )

    def has_full_time(self) -> bool:
        """Whether all time components are set."""
        return (
            self.hour is not None
            and self.minute is not None
            and self.second is not None
            and self.nanosecond is not None
        )

    def has_full_datetime(self) -> bool:
        """Whether all components are set."""
        return self.has_full_date() and self.has_full_time()

    def is_not_none(self) -> bool:
        """Check whether any component is set."""
        return self.has_date() or self.has_weekday() or self.has_time()

    def to_time(self) -> time:
        """Convert to a `datetime.time` instance.

        Fields that are None will become 0."""
        return time(
            hour=self.hour if self.hour is not None else 0,
            minute=self.minute if self.minute is not None else 0,
            second=self.second if self.second is not None else 0,
            microsecond=self.nanosecond // 1000 if self.nanosecond is not None else 0,
        )

    def to_namedtuple(self) -> namedtuple:
        """Convert to a named tuple."""
        return DTCNT(*attr.asdict(self).values())


def time_to_timedelta(t: tp.Union[tp.TimeLike, DTC], **kwargs) -> pd.Timedelta:
    """Convert a time-like object into `pd.Timedelta`."""
    if isinstance(t, str):
        t = DTC.parse_time_str(t, **kwargs)
    if isinstance(t, DTC):
        if t.has_date():
            raise ValueError("Time string has a date component")
        if t.has_weekday():
            raise ValueError("Time string has a weekday component")
        if not t.has_time():
            raise ValueError("Time string doesn't have a time component")
        t = t.to_time()

    return pd.Timedelta(
        hours=t.hour if t.hour is not None else 0,
        minutes=t.minute if t.minute is not None else 0,
        seconds=t.second if t.second is not None else 0,
        milliseconds=(t.microsecond // 1000) if t.microsecond is not None else 0,
        microseconds=(t.microsecond % 1000) if t.microsecond is not None else 0,
    )


def freq_to_timedelta64(freq: tp.FrequencyLike) -> np.timedelta64:
    """Convert a frequency-like object to `np.timedelta64`."""
    if not isinstance(freq, np.timedelta64):
        if not isinstance(freq, (pd.DateOffset, pd.Timedelta)):
            freq = freq_to_timedelta(freq)
        if isinstance(freq, pd.DateOffset):
            freq = pd.Timedelta(freq)
        freq = freq.to_timedelta64()
    if freq.dtype != np.dtype("timedelta64[ns]"):
        return freq.astype("timedelta64[ns]")
    return freq


def try_to_datetime_index(index: tp.IndexLike, parser_kwargs: tp.KwargsLike = None, **kwargs) -> tp.Index:
    """Try converting an index to a datetime index.

    `parser_kwargs` are passed to `pd.to_datetime` while `**kwargs` are passed to `dateparser.parse`.

    For defaults, see `vectorbtpro._settings.datetime`."""
    import dateparser
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    parser_kwargs = merge_dicts(datetime_cfg["parser_kwargs"], parser_kwargs)

    if not isinstance(index, pd.Index):
        if isinstance(index, str):
            try:
                index = pd.to_datetime(index, **kwargs)
                index = [index]
            except Exception as e:
                if datetime_cfg["parse_index"]:
                    try:
                        parsed_index = dateparser.parse(index, **parser_kwargs)
                        if parsed_index is None:
                            raise Exception
                        index = pd.to_datetime(parsed_index, **kwargs)
                        index = [index]
                    except Exception as e2:
                        pass
        try:
            index = pd.Index(index)
        except Exception as e:
            index = pd.Index([index])
    if isinstance(index, pd.DatetimeIndex):
        return index
    if index.dtype == object:
        try:
            return pd.to_datetime(index, **kwargs)
        except Exception as e:
            if datetime_cfg["parse_index"]:
                try:

                    def _parse(x):
                        _parsed_index = dateparser.parse(x, **parser_kwargs)
                        if _parsed_index is None:
                            raise Exception
                        return _parsed_index

                    return pd.to_datetime(index.map(_parse), **kwargs)
                except Exception as e2:
                    pass
    return index


def try_align_to_dt_index(source_index: tp.IndexLike, target_index: tp.Index, **kwargs) -> tp.Index:
    """Try aligning an index to another datetime index.

    Keyword arguments are passed to `try_to_datetime_index`."""
    source_index = try_to_datetime_index(source_index, **kwargs)
    if isinstance(source_index, pd.DatetimeIndex) and isinstance(target_index, pd.DatetimeIndex):
        if source_index.tzinfo is None and target_index.tzinfo is not None:
            source_index = source_index.tz_localize(target_index.tzinfo)
        elif source_index.tzinfo is not None and target_index.tzinfo is not None:
            source_index = source_index.tz_convert(target_index.tzinfo)
    return source_index


def try_align_dt_to_index(dt_like: tp.DatetimeLike, target_index: tp.Index, **kwargs) -> tp.DatetimeLike:
    """Try aligning a datetime-like object to another datetime index.

    Keyword arguments are passed to `to_timestamp`."""
    if not isinstance(target_index, pd.DatetimeIndex):
        return dt_like
    dt = to_timestamp(dt_like, **kwargs)
    if dt.tzinfo is None and target_index.tzinfo is not None:
        dt = dt.tz_localize(target_index.tzinfo)
    elif dt.tzinfo is not None and target_index.tzinfo is not None:
        dt = dt.tz_convert(target_index.tzinfo)
    return dt


def infer_index_freq(
    index: pd.Index,
    freq: tp.Optional[tp.FrequencyLike] = None,
    allow_date_offset: bool = True,
    allow_numeric: bool = True,
    detect_via_diff: bool = False,
) -> tp.Union[None, int, float, tp.PandasFrequency]:
    """Infer frequency of a datetime index if `freq` is None, otherwise convert `freq`."""
    if freq is None and isinstance(index, pd.DatetimeIndex):
        if index.freqstr is not None:
            freq = parse_timedelta(index.freqstr)
        elif index.freq is not None:
            freq = parse_timedelta(index.freq)
        elif len(index) >= 3:
            freq = pd.infer_freq(index)
            if freq is not None:
                freq = parse_timedelta(freq)
    if freq is None and detect_via_diff:
        return (index[1:] - index[:-1]).min()
    if freq is None:
        return None
    if checks.is_number(freq) and allow_numeric:
        return freq
    freq = parse_timedelta(freq)
    if isinstance(freq, pd.DateOffset):
        try:
            td_freq = pd.Timedelta(freq)
            if to_offset(td_freq) == freq:
                freq = td_freq
            else:
                warnings.warn(f"Ambiguous frequency {freq}", stacklevel=2)
        except Exception as e:
            if allow_date_offset:
                return freq
    return freq_to_timedelta(freq)


def get_dt_index_gaps(
    index: tp.IndexLike,
    freq: tp.Optional[tp.FrequencyLike] = None,
    skip_index: tp.Optional[tp.IndexLike] = None,
    **kwargs,
) -> tp.Tuple[tp.Index, tp.Index]:
    """Get gaps in a datetime index.

    Returns two indexes: start indexes (inclusive) and end indexes (exclusive).

    Keyword arguments are passed to `try_to_datetime_index`."""
    index = try_to_datetime_index(index, **kwargs)
    checks.assert_instance_of(index, pd.DatetimeIndex)
    if not index.is_unique:
        raise ValueError("Datetime index must be unique")
    if not index.is_monotonic_increasing:
        raise ValueError("Datetime index must be monotonically increasing")
    freq = infer_index_freq(index, freq=freq, allow_numeric=False, detect_via_diff=True)
    if skip_index is not None:
        skip_index = try_to_datetime_index(skip_index, **kwargs)
        checks.assert_instance_of(skip_index, pd.DatetimeIndex)
        skip_bound_start = skip_index.min()
        skip_bound_end = skip_index.max()
        index = index.difference(skip_index)
    else:
        skip_bound_start = None
        skip_bound_end = None
    start_index = index[:-1]
    end_index = index[1:]
    gap_mask = start_index + freq < end_index
    bound_starts = start_index[gap_mask] + freq
    bound_ends = end_index[gap_mask]
    if skip_bound_start is not None and skip_bound_start < index[0]:
        bound_starts = pd.Index([skip_bound_start]).union(bound_starts)
        bound_ends = pd.Index([index[0]]).union(bound_ends)
    if skip_bound_end is not None and skip_bound_end >= index[-1] + freq:
        bound_starts = pd.Index([index[-1] + freq]).union(bound_starts)
        bound_ends = pd.Index([skip_bound_end + freq]).union(bound_ends)
    return bound_starts, bound_ends


def get_rangebreaks(index: tp.IndexLike, **kwargs) -> list:
    """Get `rangebreaks` based on `get_dt_index_gaps`."""
    start_index, end_index = get_dt_index_gaps(index, **kwargs)
    return [dict(bounds=x) for x in zip(start_index, end_index)]


def get_utc_tz(**kwargs) -> tzinfo:
    """Get UTC timezone."""
    from dateutil.tz import tzutc

    return to_timezone(tzutc(), **kwargs)


def get_local_tz(**kwargs) -> tzinfo:
    """Get local timezone."""
    from dateutil.tz import tzlocal

    return to_timezone(tzlocal(), **kwargs)


def convert_tzaware_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """Return as non-naive time.

    `datetime.time` must have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).timetz()


def tzaware_to_naive_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """Return as naive time.

    `datetime.time` must have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time()


def naive_to_tzaware_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """Return as non-naive time.

    `datetime.time` must not have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time().replace(tzinfo=tz_out)


def convert_naive_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """Return as naive time.

    `datetime.time` must not have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time()


def is_tz_aware(dt: tp.Union[datetime, pd.Timestamp, pd.DatetimeIndex]) -> bool:
    """Whether datetime is timezone-aware."""
    tz = dt.tzinfo
    if tz is None:
        return False
    return tz.utcoffset(datetime.now()) is not None


def to_timezone(
    tz: tp.TimezoneLike,
    to_fixed_offset: tp.Optional[bool] = None,
    parser_kwargs: tp.KwargsLike = None,
) -> tzinfo:
    """Parse the timezone.

    Strings are parsed with `pandas`, and `dateparser`, while integers and floats are treated as hour offsets.

    If `to_fixed_offset` is set to True, will convert to `datetime.timezone`. See global settings.

    `parser_kwargs` are passed to `dateparser.parse`.

    For defaults, see `vectorbtpro._settings.datetime`."""
    import dateparser
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if tz is None:
        return get_local_tz()
    if to_fixed_offset is None:
        to_fixed_offset = datetime_cfg["to_fixed_offset"]
    parser_kwargs = merge_dicts(datetime_cfg["parser_kwargs"], parser_kwargs)

    if isinstance(tz, str):
        try:
            tz = pd.Timestamp("now", tz=tz).tz
        except Exception as e:
            dt = dateparser.parse("now %s" % tz, **parser_kwargs)
            if dt is not None:
                tz = dt.tzinfo
                to_fixed_offset = True
    if checks.is_number(tz):
        tz = timezone(timedelta(seconds=tz))
    if isinstance(tz, timedelta):
        tz = timezone(tz)
    if isinstance(tz, tzinfo):
        if to_fixed_offset is None:
            if tz == tz:
                to_fixed_offset = False
            else:  # Pandas has issues with this
                to_fixed_offset = True
        if to_fixed_offset:
            return timezone(tz.utcoffset(datetime.now()))
        return tz
    raise ValueError(f"Could not parse the timezone {tz}")


def to_timestamp(
    dt_like: tp.DatetimeLike,
    parser_kwargs: tp.KwargsLike = None,
    unit: str = "ns",
    tz: tp.Optional[tp.TimezoneLike] = None,
    to_fixed_offset: tp.Optional[bool] = None,
    **kwargs,
) -> pd.Timestamp:
    """Parse the datetime as a `pd.Timestamp`.

    `parser_kwargs` are passed to `pd.Timestamp` while `**kwargs` are passed to `dateparser.parse`.

    For defaults, see `vectorbtpro._settings.datetime`."""
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    parser_kwargs = merge_dicts(datetime_cfg["parser_kwargs"], parser_kwargs)
    if tz is not None:
        tz = to_timezone(tz, to_fixed_offset=to_fixed_offset, parser_kwargs=parser_kwargs)

    if isinstance(dt_like, pd.Timestamp):
        dt = dt_like
    elif checks.is_number(dt_like):
        dt = pd.Timestamp(dt_like, tz="utc", unit=unit, **kwargs)
    elif isinstance(dt_like, str):
        try:
            tz = to_timezone(dt_like.split(" ")[-1], to_fixed_offset=to_fixed_offset, parser_kwargs=parser_kwargs)
            dt_like = " ".join(dt_like.split(" ")[:-1])
        except Exception as e:
            pass
        try:
            if dt_like.lower() == "now":
                dt = pd.Timestamp.now(tz=tz)
            else:
                dt = pd.Timestamp(dt_like, **kwargs)
        except Exception as e:
            import dateparser

            settings = parser_kwargs.get("settings", {})
            settings["RELATIVE_BASE"] = settings.get("RELATIVE_BASE", pd.Timestamp.now(tz=tz).to_pydatetime())
            parser_kwargs["settings"] = settings
            dt = dateparser.parse(dt_like, **parser_kwargs)
            if dt is not None:
                if is_tz_aware(dt):
                    tz = to_timezone(dt.tzinfo, to_fixed_offset=True, parser_kwargs=parser_kwargs)
                    dt = dt.replace(tzinfo=tz)
                dt = pd.Timestamp(dt, **kwargs)
            else:
                raise ValueError(f"Could not parse the timestamp {dt_like}")
    else:
        dt = pd.Timestamp(dt_like, **kwargs)
    if tz is not None:
        if not is_tz_aware(dt):
            dt = dt.tz_localize(tz)
        else:
            dt = dt.tz_convert(tz)
    return dt


def to_tzaware_timestamp(
    dt_like: tp.DatetimeLike,
    naive_tz: tp.Optional[tp.TimezoneLike] = None,
    tz: tp.Optional[tp.TimezoneLike] = None,
    **kwargs,
) -> pd.Timestamp:
    """Parse the datetime as a timezone-aware `pd.Timestamp`.

    Uses `to_timestamp`.

    Raw timestamps are localized to UTC, while naive datetime is localized to `naive_tz`.
    Set `naive_tz` to None to use the default value defined under `vectorbtpro._settings.datetime`.
    To explicitly convert the datetime to a timezone, use `tz` (uses `to_timezone`).

    For defaults, see `vectorbtpro._settings.datetime`."""
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if naive_tz is None:
        naive_tz = datetime_cfg["naive_tz"]

    ts = to_timestamp(dt_like, tz=naive_tz, **kwargs)
    if is_tz_aware(ts):
        ts = ts.tz_localize(None).tz_localize(to_timezone(ts.tzinfo))
    if tz is not None:
        ts = ts.tz_convert(to_timezone(tz))
    return ts


def to_naive_timestamp(dt_like: tp.DatetimeLike, **kwargs) -> pd.Timestamp:
    """Parse the datetime as a timezone-naive `pd.Timestamp`."""
    return to_timestamp(dt_like, **kwargs).tz_localize(None)


def to_datetime(dt_like: tp.DatetimeLike, **kwargs) -> datetime:
    """Parse the datetime as a `datetime.datetime`.

    Uses `to_timestamp`."""
    if "unit" not in kwargs:
        kwargs["unit"] = "ms"
    return to_timestamp(dt_like, **kwargs).to_pydatetime()


def to_tzaware_datetime(dt_like: tp.DatetimeLike, **kwargs) -> datetime:
    """Parse the datetime as a timezone-aware `datetime.datetime`.

    Uses `to_tzaware_timestamp`."""
    if "unit" not in kwargs:
        kwargs["unit"] = "ms"
    return to_tzaware_timestamp(dt_like, **kwargs).to_pydatetime()


def to_naive_datetime(dt_like: tp.DatetimeLike, **kwargs) -> datetime:
    """Parse the datetime as a timezone-naive `datetime.datetime`.

    Uses `to_naive_timestamp`."""
    if "unit" not in kwargs:
        kwargs["unit"] = "ms"
    return to_naive_timestamp(dt_like, **kwargs).to_pydatetime()


def datetime_to_ms(dt: datetime) -> int:
    """Convert a datetime to milliseconds."""
    epoch = datetime.fromtimestamp(0, dt.tzinfo)
    return int((dt - epoch).total_seconds() * 1000.0)


def interval_to_ms(interval: str) -> tp.Optional[int]:
    """Convert an interval string to milliseconds."""
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }
    try:
        return int(interval[:-1]) * seconds_per_unit[interval[-1]] * 1000
    except (ValueError, KeyError):
        return None


def to_ns(obj: tp.ArrayLike) -> tp.ArrayLike:
    """Convert a datetime, timedelta, integer, or any array-like object to nanoseconds since Unix Epoch."""
    if isinstance(obj, time):
        obj = time_to_timedelta(obj)
    if isinstance(obj, pd.Timestamp):
        obj = obj.to_datetime64()
    if isinstance(obj, pd.Timedelta):
        obj = obj.to_timedelta64()
    if isinstance(obj, (datetime, date)):
        obj = np.datetime64(obj)
    if isinstance(obj, timedelta):
        obj = np.timedelta64(obj)
    if isinstance(obj, pd.DatetimeIndex):
        obj = obj.tz_localize(None).tz_localize("utc")
    if isinstance(obj, pd.PeriodIndex):
        obj = obj.to_timestamp()
    if isinstance(obj, pd.Index):
        obj = obj.values

    if not isinstance(obj, np.ndarray):
        new_obj = np.asarray(obj)
    else:
        new_obj = obj
    if np.issubdtype(new_obj.dtype, np.datetime64) and new_obj.dtype != np.dtype("datetime64[ns]"):
        new_obj = new_obj.astype("datetime64[ns]")
    if np.issubdtype(new_obj.dtype, np.timedelta64) and new_obj.dtype != np.dtype("timedelta64[ns]"):
        new_obj = new_obj.astype("timedelta64[ns]")
    new_obj = new_obj.astype(np.int64)
    if new_obj.ndim == 0 and (not isinstance(obj, np.ndarray) or obj.ndim != 0):
        return new_obj.item()
    return new_obj
