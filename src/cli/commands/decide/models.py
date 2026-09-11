"""Decide per-model overlay hooks (P4). Table-driven, no identity branches."""

# ruff: noqa: S110, SIM105
# Fail-closed try/except is this module's idiom (moved verbatim from _impl.py
# per .agents/rules/quant.md fail-closed directives); do not "fix" to logging.

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from src.portfolio.policy import PortfolioPolicy

from src.cli.commands.decide.models_overlay_hooks import (
    _hook_equity_group,
    _hook_house_money,
    _hook_mom60_abs_cash,
    _hook_mom60_concentrated,
    _hook_mom60_hold,
    _hook_mom60_peak_lock,
    _hook_mom60_raw,
)

logger = logging.getLogger(__name__)


@dataclass
class _DecideState:
    """Mutable decide-run namespace shared by orchestration and model hooks."""

    decision_date: date
    args: argparse.Namespace
    model_arg: str | None
    panel_loaded: Any
    policy: PortfolioPolicy
    master: Any
    rules: Any
    regime_str: str | None
    lev_allowed: bool | None
    inv_allowed: bool | None
    decision_weights: Any = None
    scores: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    peak_is_locked: bool = False
    house_money_is_locked: bool = False


DecideHook = Callable[[_DecideState], None]


def _hook_split_fill_allocate(state: _DecideState) -> None:
    # build decision snapshot
    snap_df = None
    try:
        import polars as _pl_p23

        panel_loaded = state.panel_loaded
        decision_date = state.decision_date
        if panel_loaded is not None and hasattr(panel_loaded, "columns"):
            if "date" in panel_loaded.columns:
                snap_df = panel_loaded.filter(_pl_p23.col("date") == decision_date)
                if snap_df.height == 0:
                    snap_df = panel_loaded
            else:
                snap_df = panel_loaded
        # if features builder available, snapshot panel is ok as raw panel with required columns
    except Exception:
        snap_df = state.panel_loaded
    try:
        if snap_df is not None and snap_df.height > 0:
            from src.alpha.base import DecisionContext
            from src.strategies.registry import STRATEGIES as _BL_P23

            try:
                ctx_p23 = DecisionContext(
                    decision_date=state.decision_date, regime=None, capital=1_000_000_000.0, held={}, rules=state.rules
                )
            except Exception:
                ctx_p23 = DecisionContext(
                    decision_date=state.decision_date,
                    regime=None,
                    capital=1_000_000_000.0,
                    held={},
                    rules=None,  # type: ignore[arg-type]
                )
            p23_model: Any = _BL_P23["sticky.split_fill_lock"]()
            scores_p23 = p23_model.score(snap_df, ctx_p23)
            if scores_p23:
                state.scores = scores_p23
            # build ADV map from snapshot trading_value or universe provider
            adv_map_p23: dict[str, float] = {}
            try:
                if "trading_value" in snap_df.columns and "ticker" in snap_df.columns:
                    for row in snap_df.iter_rows(named=True):
                        t = str(row.get("ticker"))
                        tv = row.get("trading_value")
                        if tv is not None:
                            try:
                                adv_map_p23[t] = float(tv)
                            except Exception:
                                pass
                # also try universe provider if available (not required for unit)
            except Exception:
                adv_map_p23 = {}
            part_p23 = (
                float(getattr(state.rules, "max_order_to_adv", 0.01))
                if state.rules is not None and hasattr(state.rules, "max_order_to_adv")
                else 0.01
            )
            alloc_p23 = p23_model.allocate(
                scores_p23, adv=adv_map_p23, participation=part_p23, capital=1_000_000_000.0, current_weights={}
            )
            weights_p23 = dict(getattr(alloc_p23, "weights", alloc_p23)) if alloc_p23 is not None else {}
            state.decision_weights = type("obj", (), {"weights": weights_p23, "rationale": {}})()
            state.weights = weights_p23
        else:
            try:
                state.decision_weights = state.policy.allocate(
                    state.scores,
                    regime=state.regime_str,
                    leverage_allowed=state.lev_allowed,
                    inverse_allowed=state.inv_allowed,
                )
            except TypeError:
                state.decision_weights = state.policy.allocate(state.scores)
            state.weights = state.decision_weights.weights if hasattr(state.decision_weights, "weights") else {}
    except Exception:
        try:
            state.decision_weights = state.policy.allocate(
                state.scores,
                regime=state.regime_str,
                leverage_allowed=state.lev_allowed,
                inverse_allowed=state.inv_allowed,
            )
        except TypeError:
            state.decision_weights = state.policy.allocate(state.scores)
        state.weights = state.decision_weights.weights if hasattr(state.decision_weights, "weights") else {}


def _hook_split_fill_lock(state: _DecideState) -> None:
    from src.tournament.policy import peak_lock_active

    _rules = state.rules
    try:
        _cap_est_p23 = (
            float(getattr(_rules, "initial_capital", 1_000_000_000)) if _rules is not None else 1_000_000_000.0
        )
        _init_p23 = float(getattr(_rules, "initial_capital", 1_000_000_000)) if _rules is not None else 1_000_000_000.0
        if peak_lock_active(_cap_est_p23, _init_p23, 0.40):
            state.weights = {}
            state.peak_is_locked = True
        peak_lock_active(_cap_est_p23, _init_p23, 0.40)
    except Exception:
        pass


_LIVE_STICKY_STRATEGIES: Final[frozenset[str]] = frozenset({"sticky.mom60_raw", "sticky.mom60_post_crash_anchor"})


def _sticky_live_state_name(strategy_id: str) -> str:
    return strategy_id.replace(".", "_") + "_position"


def _hook_sticky_live_allocate(state: _DecideState, *, strategy_id: str) -> None:
    if strategy_id not in _LIVE_STICKY_STRATEGIES:
        raise ValueError(f"unsupported live sticky strategy: {strategy_id}")
    from pathlib import Path

    import polars as pl

    from src.core.paths import DataPaths
    from src.core.settings import get_settings
    from src.portfolio.sizing import SizingScheme
    from src.tournament.live_decision import (
        apply_live_exposure_and_capacity_limits,
        assert_panel_input_fresh,
        assert_sleeve_inputs_fresh,
        build_live_eligible_snapshot,
        compute_live_target_weights,
        next_hold_len,
        persist_sticky_state,
        resolve_live_championship_sleeve,
        resolve_primary_ticker,
        resolve_prior_sticky_state,
        resolve_prior_trading_session,
    )

    panel = state.panel_loaded
    assert_panel_input_fresh(panel, decision_date=state.decision_date)
    snapshot = build_live_eligible_snapshot(panel, decision_date=state.decision_date)
    try:
        data_root = get_settings().data_root
        idx_path = DataPaths(root=Path(str(data_root))).silver("index_daily")
        index_daily = pl.read_parquet(str(idx_path))
    except (OSError, FileNotFoundError):
        index_daily = pl.DataFrame()
    assert_sleeve_inputs_fresh(index_daily, decision_date=state.decision_date)
    sleeve = resolve_live_championship_sleeve(index_daily, state.decision_date)
    from src.strategies.registry import STRATEGIES as _REG_P27

    model: Any = _REG_P27[strategy_id]()
    model.reset_trackers()
    data_root_for_state = get_settings().data_root
    state_path = DataPaths(root=Path(str(data_root_for_state))).state(_sticky_live_state_name(strategy_id))
    held_override = getattr(state.args, "held", None)
    if held_override:
        stripped = str(held_override).strip()
        if stripped == "" or stripped.upper() in ("CASH", "NONE"):
            held_ticker: str | None = None
            held_weight = 0.0
            prior_hold_len = 0
        else:
            held_ticker = stripped
            held_weight = 1.0
            prior_hold_len = 1
    else:
        prior_session = resolve_prior_trading_session(panel, decision_date=state.decision_date)
        held_ticker, held_weight, prior_hold_len = resolve_prior_sticky_state(state_path, prior_session=prior_session)
    if held_ticker is not None:
        model.restore_state(held=held_ticker, hold_len=prior_hold_len)
    held_map: dict[str, float] = {held_ticker: float(held_weight)} if held_ticker else {}
    rules = state.rules
    try:
        capital = float(getattr(rules, "initial_capital", 1_000_000_000))
    except (TypeError, ValueError):
        capital = 1_000_000_000.0
    intent = compute_live_target_weights(
        model,
        snapshot,
        decision_date=state.decision_date,
        held=held_map,
        capital=capital,
        rules=rules,
        championship_sleeve=sleeve,
        scheme=SizingScheme.TOP1,
        k=1,
    )
    capped_weights = apply_live_exposure_and_capacity_limits(
        intent.weights,
        panel,
        held=held_map,
        decision_date=state.decision_date,
        capital=capital,
        strategy_id=strategy_id,
    )
    state.decision_weights = intent
    state.weights = capped_weights
    if intent.kind == "cash" or not capped_weights:
        state.weights = {}
        state.peak_is_locked = True
    new_held = resolve_primary_ticker(state.weights)
    new_weight = float(state.weights.get(new_held, 0.0)) if new_held else 0.0
    new_hold_len = next_hold_len(held_ticker, prior_hold_len, new_held)
    recomputed_path = DataPaths(root=Path(str(get_settings().data_root))).state(
        _sticky_live_state_name(strategy_id)
    )
    persist_sticky_state(
        recomputed_path, decision_date=state.decision_date, held=new_held, held_weight=new_weight, hold_len=new_hold_len
    )


def _hook_mom60_raw_allocate(state: _DecideState) -> None:
    _hook_sticky_live_allocate(state, strategy_id="sticky.mom60_raw")


def _hook_mom60_post_crash_anchor_allocate(state: _DecideState) -> None:
    _hook_sticky_live_allocate(state, strategy_id="sticky.mom60_post_crash_anchor")


_ALLOCATE_HOOKS: Final[dict[str, Any]] = {
    "sticky.split_fill_lock": _hook_split_fill_allocate,
    "sticky.mom60_raw": _hook_mom60_raw_allocate,
    "sticky.mom60_post_crash_anchor": _hook_mom60_post_crash_anchor_allocate,
}

_OVERLAY_HOOKS: Final[dict[str, Any]] = {
    "sticky.split_fill_lock": _hook_split_fill_lock,
    "sticky.mom60_peak_lock": _hook_mom60_peak_lock,
    "sticky.house_money": _hook_house_money,
    "sticky.mom60_concentrated": _hook_mom60_concentrated,
    "sticky.mom60_raw": _hook_mom60_raw,
    "sticky.mom60_post_crash_anchor": _hook_mom60_raw,
    "sticky.mom60_hold": _hook_mom60_hold,
    "sticky.mom60_abs_cash": _hook_mom60_abs_cash,
    "sticky.equity_mom60": _hook_equity_group,
    "sticky.equity_mom60_vol": _hook_equity_group,
    "sticky.fillable_mom60": _hook_equity_group,
    "convex.lottery_impulse": _hook_equity_group,
    "sticky.mom60_runner_reversal": _hook_equity_group,
}

__all__ = ["_ALLOCATE_HOOKS", "_OVERLAY_HOOKS", "DecideHook", "_DecideState"]
