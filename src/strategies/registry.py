from __future__ import annotations

import contextlib
import difflib
from collections.abc import Callable, Mapping
from typing import Any, Final, cast

from src.strategies.ids import (
    ALPHA_FAMILY_INTENSITY,
    ALPHA_SECTOR_LEADERSHIP,
    BASELINE_BUY_HOLD,
    BASELINE_MOM20_EQUAL3,
    BASELINE_MOM20_MA_GATE,
    BASELINE_MOM20_TOP1,
    BASELINE_REGIME_GATED_THEME,
    BASELINE_THEME_MOMENTUM,
    CHAMPION_TAIL_RANKER,
    CONVEX_LOTTERY_IMPULSE,
    PORTFOLIO_CONVEXITY_HOLD,
    PORTFOLIO_CONVEXITY_REBALANCE,
    PORTFOLIO_CONVEXITY_VARIANT,
    PORTFOLIO_LEADERSHIP_CONFIDENCE,
    PORTFOLIO_LEADERSHIP_POLICY,
    PORTFOLIO_LOTTERY_EXPOSURE,
    PORTFOLIO_LOTTERY_REBALANCE,
    PORTFOLIO_MOMENTUM_CONFIDENCE,
    PORTFOLIO_MOMENTUM_POLICY,
    PORTFOLIO_MOMENTUM_VEHICLE,
    PORTFOLIO_TAIL_CONCENTRATION,
    STICKY_EQUITY_MOM60,
    STICKY_EQUITY_MOM60_VOL,
    STICKY_FAMILY_PEAK_LOCK,
    STICKY_FILLABLE_MOM60,
    STICKY_HOUSE_MONEY,
    STICKY_IMPULSE_CRASH,
    STICKY_LEADER_BASE,
    STICKY_MOM60_ABS_CASH,
    STICKY_MOM60_CONCENTRATED,
    STICKY_MOM60_HOLD,
    STICKY_MOM60_INACTIVE_PARTICIPATE,
    STICKY_MOM60_PEAK_LOCK,
    STICKY_MOM60_POST_CRASH_ANCHOR,
    STICKY_MOM60_RAW,
    STICKY_MOM60_RUNNER_REVERSAL,
    STICKY_P27_COMPLEMENT_SWITCH,
    STICKY_SPLIT_FILL_LOCK,
)
from src.strategies.protocol import StrategyProtocol

_ALL_SEMANTIC: Final[frozenset[str]] = frozenset(
    {
        BASELINE_BUY_HOLD,
        BASELINE_MOM20_TOP1,
        BASELINE_MOM20_EQUAL3,
        BASELINE_MOM20_MA_GATE,
        BASELINE_THEME_MOMENTUM,
        BASELINE_REGIME_GATED_THEME,
        ALPHA_SECTOR_LEADERSHIP,
        ALPHA_FAMILY_INTENSITY,
        PORTFOLIO_MOMENTUM_POLICY,
        PORTFOLIO_MOMENTUM_VEHICLE,
        PORTFOLIO_MOMENTUM_CONFIDENCE,
        PORTFOLIO_LEADERSHIP_POLICY,
        PORTFOLIO_LEADERSHIP_CONFIDENCE,
        PORTFOLIO_LOTTERY_EXPOSURE,
        PORTFOLIO_TAIL_CONCENTRATION,
        PORTFOLIO_CONVEXITY_HOLD,
        PORTFOLIO_CONVEXITY_REBALANCE,
        PORTFOLIO_CONVEXITY_VARIANT,
        PORTFOLIO_LOTTERY_REBALANCE,
        STICKY_LEADER_BASE,
        STICKY_IMPULSE_CRASH,
        STICKY_FAMILY_PEAK_LOCK,
        STICKY_SPLIT_FILL_LOCK,
        STICKY_MOM60_PEAK_LOCK,
        STICKY_HOUSE_MONEY,
        STICKY_MOM60_CONCENTRATED,
        STICKY_MOM60_RAW,
        STICKY_P27_COMPLEMENT_SWITCH,
        STICKY_MOM60_HOLD,
        STICKY_MOM60_ABS_CASH,
        STICKY_EQUITY_MOM60,
        STICKY_EQUITY_MOM60_VOL,
        STICKY_FILLABLE_MOM60,
        CONVEX_LOTTERY_IMPULSE,
        STICKY_MOM60_RUNNER_REVERSAL,
        STICKY_MOM60_INACTIVE_PARTICIPATE,
        STICKY_MOM60_POST_CRASH_ANCHOR,
        CHAMPION_TAIL_RANKER,
    }
)


def resolve_strategy_id(key: str) -> str:
    """Resolve a semantic strategy id; legacy P/B/M codes are rejected (D1).

    Accepts exact semantic ids, case-insensitively (`STICKY.MOM60_RAW`).
    Unknown keys raise ValueError listing the closest semantic ids.
    """
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"unknown strategy: {key!r}")
    k = key.strip()
    if k in _ALL_SEMANTIC:
        return k
    lowered = k.lower()
    for semantic in _ALL_SEMANTIC:
        if semantic.lower() == lowered:
            return semantic
    suggestions = difflib.get_close_matches(lowered, sorted(_ALL_SEMANTIC), n=3, cutoff=0.4)
    hint = f" did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ValueError(f"unknown strategy: {key!r}.{hint}")


def build_strategy_registry() -> Mapping[str, Callable[[], StrategyProtocol]]:
    from src.portfolio.builders_convexity import (
        make_portfolio_convexity_hold,
        make_portfolio_convexity_rebalance,
        make_portfolio_convexity_variant,
        make_portfolio_lottery_exposure,
        make_portfolio_lottery_rebalance,
        make_portfolio_tail_concentration,
    )
    from src.portfolio.builders_leadership import (
        make_portfolio_leadership_confidence,
        make_portfolio_leadership_policy,
    )
    from src.portfolio.builders_momentum import (
        make_portfolio_momentum_confidence,
        make_portfolio_momentum_policy,
        make_portfolio_momentum_vehicle,
    )
    from src.strategies.champion_tail import ChampionTailPolicy
    from src.strategies.factories.alpha import make_alpha_family_intensity, make_alpha_sector_leadership
    from src.strategies.factories.baselines import (
        make_baseline_buy_hold,
        make_baseline_mom20_equal3,
        make_baseline_mom20_ma_gate,
        make_baseline_mom20_top1,
        make_baseline_regime_gated_theme,
        make_baseline_theme_momentum,
    )
    from src.strategies.factories.convex import make_convex_lottery_impulse
    from src.strategies.factories.sticky import (
        make_sticky_equity_mom60,
        make_sticky_equity_mom60_vol,
        make_sticky_family_peak_lock,
        make_sticky_fillable_mom60,
        make_sticky_house_money,
        make_sticky_impulse_crash,
        make_sticky_leader_base,
        make_sticky_mom60_abs_cash,
        make_sticky_mom60_concentrated,
        make_sticky_mom60_hold,
        make_sticky_mom60_inactive_participate,
        make_sticky_mom60_peak_lock,
        make_sticky_mom60_post_crash_anchor,
        make_sticky_mom60_raw,
        make_sticky_mom60_runner_reversal,
        make_sticky_split_fill_lock,
    )
    from src.tournament.p27_complement_sleeve import make_p27_complement_switch

    raw: dict[str, Callable[[], Any]] = {
        BASELINE_BUY_HOLD: make_baseline_buy_hold,
        BASELINE_MOM20_TOP1: make_baseline_mom20_top1,
        BASELINE_MOM20_EQUAL3: make_baseline_mom20_equal3,
        BASELINE_MOM20_MA_GATE: make_baseline_mom20_ma_gate,
        BASELINE_THEME_MOMENTUM: make_baseline_theme_momentum,
        BASELINE_REGIME_GATED_THEME: make_baseline_regime_gated_theme,
        ALPHA_SECTOR_LEADERSHIP: make_alpha_sector_leadership,
        ALPHA_FAMILY_INTENSITY: make_alpha_family_intensity,
        PORTFOLIO_MOMENTUM_POLICY: make_portfolio_momentum_policy,
        PORTFOLIO_MOMENTUM_VEHICLE: make_portfolio_momentum_vehicle,
        PORTFOLIO_MOMENTUM_CONFIDENCE: make_portfolio_momentum_confidence,
        PORTFOLIO_LEADERSHIP_POLICY: make_portfolio_leadership_policy,
        PORTFOLIO_LEADERSHIP_CONFIDENCE: make_portfolio_leadership_confidence,
        PORTFOLIO_LOTTERY_EXPOSURE: make_portfolio_lottery_exposure,
        PORTFOLIO_TAIL_CONCENTRATION: make_portfolio_tail_concentration,
        PORTFOLIO_CONVEXITY_HOLD: make_portfolio_convexity_hold,
        PORTFOLIO_CONVEXITY_REBALANCE: make_portfolio_convexity_rebalance,
        PORTFOLIO_CONVEXITY_VARIANT: make_portfolio_convexity_variant,
        PORTFOLIO_LOTTERY_REBALANCE: make_portfolio_lottery_rebalance,
        STICKY_LEADER_BASE: make_sticky_leader_base,
        STICKY_IMPULSE_CRASH: make_sticky_impulse_crash,
        STICKY_FAMILY_PEAK_LOCK: make_sticky_family_peak_lock,
        STICKY_SPLIT_FILL_LOCK: make_sticky_split_fill_lock,
        STICKY_MOM60_PEAK_LOCK: make_sticky_mom60_peak_lock,
        STICKY_HOUSE_MONEY: make_sticky_house_money,
        STICKY_MOM60_CONCENTRATED: make_sticky_mom60_concentrated,
        STICKY_MOM60_RAW: make_sticky_mom60_raw,
        STICKY_MOM60_HOLD: make_sticky_mom60_hold,
        STICKY_MOM60_ABS_CASH: make_sticky_mom60_abs_cash,
        STICKY_EQUITY_MOM60: make_sticky_equity_mom60,
        STICKY_EQUITY_MOM60_VOL: make_sticky_equity_mom60_vol,
        STICKY_FILLABLE_MOM60: make_sticky_fillable_mom60,
        STICKY_MOM60_RUNNER_REVERSAL: make_sticky_mom60_runner_reversal,
        STICKY_MOM60_INACTIVE_PARTICIPATE: make_sticky_mom60_inactive_participate,
        STICKY_MOM60_POST_CRASH_ANCHOR: make_sticky_mom60_post_crash_anchor,
        CONVEX_LOTTERY_IMPULSE: make_convex_lottery_impulse,
        CHAMPION_TAIL_RANKER: lambda: ChampionTailPolicy(),
    }
    raw[STICKY_P27_COMPLEMENT_SWITCH] = make_p27_complement_switch

    def _wrap(semantic: str, factory: Callable[[], Any]) -> Callable[[], StrategyProtocol]:
        def _factory() -> StrategyProtocol:
            obj = cast(StrategyProtocol, factory())
            with contextlib.suppress(Exception):
                obj.name = semantic
            return obj

        return _factory

    for key in raw:
        resolve_strategy_id(key)
    return {semantic: _wrap(semantic, factory) for semantic, factory in raw.items()}


STRATEGIES: Final[Mapping[str, Callable[[], StrategyProtocol]]] = build_strategy_registry()


def family_of(strategy_id: str) -> str:
    """Return the semantic family prefix (the part before the dot)."""
    try:
        sem = resolve_strategy_id(strategy_id)
    except Exception:
        sem = str(strategy_id)
    if "." in sem:
        return sem.split(".", 1)[0]
    return "unknown"
