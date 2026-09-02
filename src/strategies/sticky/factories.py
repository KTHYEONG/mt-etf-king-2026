# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

from collections.abc import Callable, Mapping

from src.alpha.baselines import (
    _make_p20,
    _make_p21,
    _make_p22,
    _make_p23,
    _make_p24,
    _make_p25,
    _make_p26,
    _make_p27,
    _make_p28a,
    _make_p28b,
)
from src.strategies.ids import (
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
}

__all__ = ["make_sticky_mom60_raw", "FACTORY_REGISTRY"]
