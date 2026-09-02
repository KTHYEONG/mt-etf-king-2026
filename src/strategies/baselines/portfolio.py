# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

from src.alpha.baselines import (
    _make_b0,
    _make_b1,
    _make_b2,
    _make_b3,
    _make_b4,
    _make_b5,
    _make_m07,
    _make_m13,
    _make_p08,
    _make_p10,
    _make_p11,
    _make_p12,
    _make_p13,
    _make_p14,
    _make_p15,
    _make_p16,
    _make_p17,
    _make_p18,
    _make_p19,
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

def make_portfolio_momentum_policy() -> object:
    return _make_p13()


__all__ = ["_make_p08", "_make_p10", "_make_p11", "_make_p12", "_make_p13", "_make_p14", "_make_p15", "_make_p16", "_make_p17", "_make_p18", "_make_p19", "make_portfolio_momentum_policy"]
