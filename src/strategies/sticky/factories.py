# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

from collections.abc import Callable, Mapping

from src.strategies.factories.sticky import (
    make_sticky_equity_mom60 as _make_p29,
    make_sticky_equity_mom60_vol as _make_p29v,
    make_sticky_family_peak_lock as _make_p22,
    make_sticky_fillable_mom60 as _make_p30,
    make_sticky_house_money as _make_p25,
    make_sticky_impulse_crash as _make_p21,
    make_sticky_leader_base as _make_p20,
    make_sticky_mom60_abs_cash as _make_p28b,
    make_sticky_mom60_concentrated as _make_p26,
    make_sticky_mom60_hold as _make_p28a,
    make_sticky_mom60_peak_lock as _make_p24,
    make_sticky_mom60_raw as _make_p27,
    make_sticky_split_fill_lock as _make_p23,
)
from src.strategies.ids import (
    STICKY_EQUITY_MOM60,
    STICKY_EQUITY_MOM60_VOL,
    STICKY_FILLABLE_MOM60,
    STICKY_FAMILY_PEAK_LOCK,
    STICKY_HOUSE_MONEY,
    STICKY_IMPULSE_CRASH,
    STICKY_LEADER_BASE,
    STICKY_MOM60_ABS_CASH,
    STICKY_MOM60_CONCENTRATED,
    STICKY_MOM60_HOLD,
    STICKY_MOM60_PEAK_LOCK,
    STICKY_MOM60_RAW,
    STICKY_SPLIT_FILL_LOCK,
)
from src.strategies.sticky.model import StickyLeaderModel



def make_sticky_mom60_raw() -> object:
    return _make_p27()


FACTORY_REGISTRY: Mapping[str, Callable[[], object]] = {
    STICKY_LEADER_BASE: _make_p20,
    STICKY_IMPULSE_CRASH: _make_p21,
    STICKY_FAMILY_PEAK_LOCK: _make_p22,
    STICKY_SPLIT_FILL_LOCK: _make_p23,
    STICKY_MOM60_PEAK_LOCK: _make_p24,
    STICKY_HOUSE_MONEY: _make_p25,
    STICKY_MOM60_CONCENTRATED: _make_p26,
    STICKY_MOM60_RAW: _make_p27,
    STICKY_MOM60_HOLD: _make_p28a,
    STICKY_MOM60_ABS_CASH: _make_p28b,
    STICKY_EQUITY_MOM60: _make_p29,
    STICKY_EQUITY_MOM60_VOL: _make_p29v,
    STICKY_FILLABLE_MOM60: _make_p30,
}

__all__ = ["make_sticky_mom60_raw", "FACTORY_REGISTRY"]
