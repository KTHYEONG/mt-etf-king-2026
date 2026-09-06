# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

from src.portfolio.builders_convexity import (
    make_portfolio_convexity_hold as _make_p16,
    make_portfolio_convexity_rebalance as _make_p17,
    make_portfolio_convexity_variant as _make_p18,
    make_portfolio_lottery_rebalance as _make_p19,
    make_portfolio_tail_concentration as _make_p15,
)
from src.portfolio.builders_leadership import (
    make_portfolio_leadership_confidence as _make_p13,
    make_portfolio_leadership_policy as _make_p12,
)
from src.portfolio.builders_momentum import (
    make_portfolio_momentum_confidence as _make_p11,
    make_portfolio_momentum_policy as _make_p08,
    make_portfolio_momentum_vehicle as _make_p10,
)
from src.portfolio.builders_convexity import make_portfolio_lottery_exposure as _make_p14
from src.strategies.factories.alpha import (
    make_alpha_family_intensity as _make_m13,
    make_alpha_sector_leadership as _make_m07,
)
from src.strategies.factories.baselines import (
    make_baseline_buy_hold as _make_b0,
    make_baseline_mom20_equal3 as _make_b2,
    make_baseline_mom20_ma_gate as _make_b3,
    make_baseline_mom20_top1 as _make_b1,
    make_baseline_regime_gated_theme as _make_b5,
    make_baseline_theme_momentum as _make_b4,
)
from src.strategies.factories.sticky import (
    make_sticky_family_peak_lock as _make_p22,
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


def make_portfolio_momentum_policy() -> object:
    return _make_p13()


__all__ = ["_make_p08", "_make_p10", "_make_p11", "_make_p12", "_make_p13", "_make_p14", "_make_p15", "_make_p16", "_make_p17", "_make_p18", "_make_p19", "make_portfolio_momentum_policy"]
