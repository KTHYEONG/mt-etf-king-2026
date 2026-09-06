# mypy: ignore-errors
# ruff: noqa
"""Sticky-leader strategy factories facade (P5: bodies live in sticky_core/sticky_locks/sticky_mom60)."""

from __future__ import annotations

from src.strategies.factories.sticky_core import (
    make_sticky_family_peak_lock,
    make_sticky_impulse_crash,
    make_sticky_leader_base,
)
from src.strategies.factories.sticky_locks import (
    make_sticky_house_money,
    make_sticky_mom60_peak_lock,
    make_sticky_split_fill_lock,
)
from src.strategies.factories.sticky_mom60 import (
    make_sticky_equity_mom60,
    make_sticky_equity_mom60_vol,
    make_sticky_fillable_mom60,
    make_sticky_mom60_abs_cash,
    make_sticky_mom60_concentrated,
    make_sticky_mom60_hold,
    make_sticky_mom60_raw,
    make_sticky_mom60_runner_reversal,
)

__all__ = [
    "make_sticky_equity_mom60",
    "make_sticky_equity_mom60_vol",
    "make_sticky_family_peak_lock",
    "make_sticky_fillable_mom60",
    "make_sticky_house_money",
    "make_sticky_impulse_crash",
    "make_sticky_leader_base",
    "make_sticky_mom60_abs_cash",
    "make_sticky_mom60_concentrated",
    "make_sticky_mom60_hold",
    "make_sticky_mom60_peak_lock",
    "make_sticky_mom60_raw",
    "make_sticky_mom60_runner_reversal",
    "make_sticky_split_fill_lock",
]
