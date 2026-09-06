# mypy: ignore-errors
# ruff: noqa
"""Buy-and-hold and momentum baseline strategy factories (P3 split of src/alpha/baselines.py)."""

from __future__ import annotations

from src.features.regime import RegimeState
from src.strategies.baselines.core import BuyAndHoldBaseline
from src.strategies.factories._shared import (
    MomentumTrendFilter,
    RegimeGatedMomentum,
    ThemeMomentum,
    TopKMomentum,
)


def make_baseline_buy_hold() -> BuyAndHoldBaseline:
    # Default ticker for buy-and-hold: KODEX 200 = 069500
    return BuyAndHoldBaseline(ticker="069500", name="baseline.buy_hold")


def make_baseline_mom20_top1() -> TopKMomentum:
    return TopKMomentum(horizon=20, name="baseline.mom20_top1")


def make_baseline_mom20_equal3() -> TopKMomentum:
    # B2 is Top-3 equal weighted; same scoring as B1 but sizing differs via config. For baseline registry, we provide same model with name B2
    return TopKMomentum(horizon=20, name="baseline.mom20_equal3")


def make_baseline_mom20_ma_gate() -> MomentumTrendFilter:
    return MomentumTrendFilter(horizon=20, ma_window=20, name="baseline.mom20_ma_gate")


def make_baseline_theme_momentum() -> ThemeMomentum:
    return ThemeMomentum(horizon=20, name="baseline.theme_momentum")


def make_baseline_regime_gated_theme() -> RegimeGatedMomentum:
    inner = ThemeMomentum(horizon=20, name="B5_inner")
    blocked = frozenset({RegimeState.STRONG_RISK_OFF})
    return RegimeGatedMomentum(inner=inner, blocked=blocked, name="baseline.regime_gated_theme")
