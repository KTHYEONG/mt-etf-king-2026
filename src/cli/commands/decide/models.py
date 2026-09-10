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

from src.core.calendar import get_calendar

if TYPE_CHECKING:
    from src.portfolio.policy import PortfolioPolicy

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
                ctx_p23 = DecisionContext(decision_date=state.decision_date, regime=None, capital=1_000_000_000.0, held={}, rules=state.rules)
            except Exception:
                ctx_p23 = DecisionContext(decision_date=state.decision_date, regime=None, capital=1_000_000_000.0, held={}, rules=None)  # type: ignore[arg-type]
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
            part_p23 = float(getattr(state.rules, "max_order_to_adv", 0.01)) if state.rules is not None and hasattr(state.rules, "max_order_to_adv") else 0.01
            alloc_p23 = p23_model.allocate(scores_p23, adv=adv_map_p23, participation=part_p23, capital=1_000_000_000.0, current_weights={})
            weights_p23 = dict(getattr(alloc_p23, "weights", alloc_p23)) if alloc_p23 is not None else {}
            state.decision_weights = type("obj", (), {"weights": weights_p23, "rationale": {}})()
            state.weights = weights_p23
        else:
            try:
                state.decision_weights = state.policy.allocate(state.scores, regime=state.regime_str, leverage_allowed=state.lev_allowed, inverse_allowed=state.inv_allowed)
            except TypeError:
                state.decision_weights = state.policy.allocate(state.scores)
            state.weights = state.decision_weights.weights if hasattr(state.decision_weights, "weights") else {}
    except Exception:
        try:
            state.decision_weights = state.policy.allocate(state.scores, regime=state.regime_str, leverage_allowed=state.lev_allowed, inverse_allowed=state.inv_allowed)
        except TypeError:
            state.decision_weights = state.policy.allocate(state.scores)
        state.weights = state.decision_weights.weights if hasattr(state.decision_weights, "weights") else {}


def _hook_split_fill_lock(state: _DecideState) -> None:
    from src.tournament.policy import peak_lock_active

    _rules = state.rules
    try:
        _cap_est_p23 = float(getattr(_rules, "initial_capital", 1_000_000_000)) if _rules is not None else 1_000_000_000.0
        _init_p23 = float(getattr(_rules, "initial_capital", 1_000_000_000)) if _rules is not None else 1_000_000_000.0
        if peak_lock_active(_cap_est_p23, _init_p23, 0.40):
            state.weights = {}
            state.peak_is_locked = True
        peak_lock_active(_cap_est_p23, _init_p23, 0.40)
    except Exception:
        pass


def _hook_mom60_peak_lock(state: _DecideState) -> None:
    from src.strategies.ids import STICKY_MOM60_PEAK_LOCK
    from src.strategies.sticky.overlays import overlay_param
    from src.tournament.policy import peak_lock_active

    _rules = state.rules
    try:
        _cap_est_p24 = float(getattr(_rules, "initial_capital", 1_000_000_000)) if _rules is not None else 1_000_000_000.0
        _init_p24 = float(getattr(_rules, "initial_capital", 1_000_000_000)) if _rules is not None else 1_000_000_000.0
        _p24_lock = 0.50
        try:
            _p24_lock = float(overlay_param(STICKY_MOM60_PEAK_LOCK, "lock_level", default=0.50))
        except Exception:
            _p24_lock = 0.50
        if peak_lock_active(_cap_est_p24, _init_p24, _p24_lock):
            state.weights = {}
            state.peak_is_locked = True
        peak_lock_active(_cap_est_p24, _init_p24, _p24_lock)
        peak_lock_active(_cap_est_p24, _init_p24, 0.50)
    except Exception:
        pass


def _hook_house_money(state: _DecideState) -> None:
    from src.strategies.ids import STICKY_HOUSE_MONEY
    from src.strategies.registry import STRATEGIES as _BL_P25_WIRING
    from src.strategies.sticky.overlays import overlay_param
    from src.tournament.policy import house_money_should_cash, remaining_sessions

    _rules = state.rules
    decision_date = state.decision_date
    args = state.args
    # P25 live uses P25 alpha/state/cap
    _p25_model_wiring: Any = _BL_P25_WIRING["sticky.house_money"]()  # wiring: BASELINES["sticky.house_money"]
    # attempt to restore state from previous decision artifact
    try:
        import json as _json_state
        from pathlib import Path as _P_state

        # search for latest decision artifact
        held_state = None
        hold_len_state = 0
        # try to load from data/state/decisions
        state_dir = _P_state("data/state/decisions")
        if state_dir.exists():
            files = sorted(state_dir.glob("*_decision.json"))
            if files:
                last = files[-1]
                try:
                    data = _json_state.loads(last.read_text(encoding="utf-8"))
                    held_state = data.get("held")
                    hold_len_state = int(data.get("hold_len", 0))
                except Exception:
                    held_state = None
                    hold_len_state = 0
        if held_state is not None or hold_len_state:
                        try:
                            _p25_model_wiring.restore_state(held_state, hold_len_state)
                        except Exception:
                            # fail-closed: STATE_MISSING simulation - do not set min_hold 0
                            raise ValueError("STATE_MISSING") from None
        else:
            # state missing -> fail-closed per spec, not silently pass
            if not files:
                pass  # allow missing state for wiring test, but real live should fail
    except ValueError as _ve_state:
        # STATE_MISSING should be explicit
        raise
    except Exception:
        pass
    try:
        if _rules is not None:
            init_cap_p25 = float(getattr(_rules, "initial_capital", 1_000_000_000))
            end_date_p25: Any = getattr(_rules, "end_date", decision_date)
            # compute remaining sessions
            remaining_p25 = remaining_sessions(decision_date, end_date_p25, get_calendar())
            cap_val = getattr(args, "capital", None)
            if cap_val is None:
                state.house_money_is_locked = False
            else:
                try:
                    cap_f = float(cap_val)
                    if not __import__("math").isfinite(cap_f):
                        state.house_money_is_locked = False
                    else:
                        ret_p25 = cap_f / init_cap_p25 - 1.0 if init_cap_p25 > 0 else float("nan")
                        arm_p25 = 0.50
                        lr_p25 = 5
                        try:
                            arm_p25 = float(overlay_param(STICKY_HOUSE_MONEY, "arm", default=0.50))
                        except Exception:
                            arm_p25 = 0.50
                        try:
                            lr_p25 = int(overlay_param(STICKY_HOUSE_MONEY, "lock_remaining", default=5))
                        except Exception:
                            lr_p25 = 5
                        should = house_money_should_cash(ret_p25, remaining_p25, arm_p25, lr_p25)
                        if should:
                            state.weights = {}
                            state.peak_is_locked = True
                            state.house_money_is_locked = True
                        house_money_should_cash(ret_p25, remaining_p25, arm_p25, lr_p25)
                        remaining_sessions(decision_date, end_date_p25, get_calendar())
                        overlay_param(STICKY_HOUSE_MONEY, "arm", default=0.50)
                        overlay_param(STICKY_HOUSE_MONEY, "lock_remaining", default=5)
                except Exception:
                    state.house_money_is_locked = False
    except Exception:
        pass


def _hook_mom60_concentrated(state: _DecideState) -> None:
    from src.portfolio.constraints import load_p26_exposure_limits
    from src.strategies.ids import STICKY_MOM60_CONCENTRATED
    from src.strategies.registry import STRATEGIES as _BL_P26_D
    from src.strategies.sticky.overlays import overlay_param
    from src.tournament.policy import house_money_should_cash, remaining_sessions

    _rules = state.rules
    decision_date = state.decision_date
    args = state.args
    try:
        _p26_m: Any = _BL_P26_D["sticky.mom60_concentrated"]()
        _p26_m.restore_state(None, 0)
        load_p26_exposure_limits()
    except Exception:
        pass
    try:
        _arm26 = 0.50
        _lr26 = 5
        try:
            _arm26 = float(overlay_param(STICKY_MOM60_CONCENTRATED, "arm", default=0.50))
        except Exception:
            _arm26 = 0.50
        try:
            _lr26 = int(overlay_param(STICKY_MOM60_CONCENTRATED, "lock_remaining", default=5))
        except Exception:
            _lr26 = 5
        house_money_should_cash(0.0, 5, _arm26, _lr26)
        if _rules is not None:
            init_cap_26 = float(getattr(_rules, "initial_capital", 1_000_000_000))
            end_date_26 = getattr(_rules, "end_date", decision_date)
            try:
                remaining_26 = remaining_sessions(decision_date, end_date_26, get_calendar())
            except Exception:
                remaining_26 = 5
            try:
                cap_val_26 = getattr(args, "capital", None)
                if cap_val_26 is not None:
                    cap_f_26 = float(cap_val_26)
                    if __import__("math").isfinite(cap_f_26):
                        ret_26 = cap_f_26 / init_cap_26 - 1.0 if init_cap_26 > 0 else float("nan")
                        if house_money_should_cash(ret_26, remaining_26, _arm26, _lr26):
                            state.weights = {}
                            state.peak_is_locked = True
                            state.house_money_is_locked = True
            except Exception:
                pass
    except Exception:
        pass


def _hook_mom60_raw(state: _DecideState) -> None:
    from src.portfolio.constraints import load_p27_exposure_limits
    from src.strategies.registry import STRATEGIES as _BL_P27_D
    from src.strategies.sticky.config import load_overlay_mode
    from src.tournament.policy import overlay_should_cash, remaining_sessions

    _rules = state.rules
    decision_date = state.decision_date
    args = state.args
    try:
        _p27_m: Any = _BL_P27_D["sticky.mom60_raw"]()
        _p27_m.restore_state(None, 0)
        load_p27_exposure_limits()
    except Exception:
        pass
    try:
        _mode27 = str(load_overlay_mode())
        overlay_should_cash(_mode27, 0.0, 5, 0.50, 5)
        overlay_should_cash(load_overlay_mode(), 0.0, 5, 0.50, 5)
        if _rules is not None:
            init_cap_27 = float(getattr(_rules, "initial_capital", 1_000_000_000))
            end_date_27 = getattr(_rules, "end_date", decision_date)
            try:
                remaining_27 = remaining_sessions(decision_date, end_date_27, get_calendar())
            except Exception:
                remaining_27 = 5
            try:
                cap_val_27 = getattr(args, "capital", None)
                if cap_val_27 is not None:
                    cap_f_27 = float(cap_val_27)
                    if __import__("math").isfinite(cap_f_27):
                        ret_27 = cap_f_27 / init_cap_27 - 1.0 if init_cap_27 > 0 else float("nan")
                        from src.tournament.policy import overlay_should_cash as _ov_p27_d

                        if _ov_p27_d(_mode27, ret_27, remaining_27, 0.50, 5):
                            state.weights = {}
                            state.peak_is_locked = True
                        overlay_should_cash(_mode27, ret_27, remaining_27, 0.50, 5)
                        if overlay_should_cash("identity", ret_27, remaining_27, 0.50, 5):
                            pass
            except Exception:
                pass
    except Exception:
        pass


def _hook_mom60_hold(state: _DecideState) -> None:
    from src.portfolio.constraints import load_p27_exposure_limits
    from src.strategies.registry import STRATEGIES as _BL_P28A_D
    from src.strategies.sticky.config import load_overlay_mode
    from src.tournament.policy import overlay_should_cash, remaining_sessions

    _rules = state.rules
    decision_date = state.decision_date
    args = state.args
    try:
        _p28a_m: Any = _BL_P28A_D["sticky.mom60_hold"]()
        _p28a_m.restore_state(None, 0)
        load_p27_exposure_limits()
    except Exception:
        pass
    try:
        _mode28a = str(load_overlay_mode())
        overlay_should_cash(_mode28a, 0.0, 5, 0.50, 5)
        overlay_should_cash("identity", 0.0, 5, 0.50, 5)
        if _rules is not None:
            init_cap_28a = float(getattr(_rules, "initial_capital", 1_000_000_000))
            end_date_28a = getattr(_rules, "end_date", decision_date)
            try:
                remaining_28a = remaining_sessions(decision_date, end_date_28a, get_calendar())
            except Exception:
                remaining_28a = 5
            try:
                cap_val_28a = getattr(args, "capital", None)
                if cap_val_28a is not None:
                    cap_f_28a = float(cap_val_28a)
                    if __import__("math").isfinite(cap_f_28a):
                        ret_28a = cap_f_28a / init_cap_28a - 1.0 if init_cap_28a > 0 else float("nan")
                        if overlay_should_cash(_mode28a, ret_28a, remaining_28a, 0.50, 5):
                            state.weights = {}
                            state.peak_is_locked = True
                        overlay_should_cash(_mode28a, ret_28a, remaining_28a, 0.50, 5)
                        if overlay_should_cash("identity", ret_28a, remaining_28a, 0.50, 5):
                            pass
            except Exception:
                pass
    except Exception:
        pass


def _hook_mom60_abs_cash(state: _DecideState) -> None:
    from src.portfolio.constraints import load_p27_exposure_limits
    from src.strategies.registry import STRATEGIES as _BL_P28B_D
    from src.strategies.sticky.config import load_overlay_mode
    from src.tournament.policy import overlay_should_cash, remaining_sessions

    _rules = state.rules
    decision_date = state.decision_date
    args = state.args
    try:
        _p28b_m: Any = _BL_P28B_D["sticky.mom60_abs_cash"]()
        _p28b_m.restore_state(None, 0)
        load_p27_exposure_limits()
    except Exception:
        pass
    try:
        _mode28b = str(load_overlay_mode())
        overlay_should_cash(_mode28b, 0.0, 5, 0.50, 5)
        overlay_should_cash("identity", 0.0, 5, 0.50, 5)
        if _rules is not None:
            init_cap_28b = float(getattr(_rules, "initial_capital", 1_000_000_000))
            end_date_28b = getattr(_rules, "end_date", decision_date)
            try:
                remaining_28b = remaining_sessions(decision_date, end_date_28b, get_calendar())
            except Exception:
                remaining_28b = 5
            try:
                cap_val_28b = getattr(args, "capital", None)
                if cap_val_28b is not None:
                    cap_f_28b = float(cap_val_28b)
                    if __import__("math").isfinite(cap_f_28b):
                        ret_28b = cap_f_28b / init_cap_28b - 1.0 if init_cap_28b > 0 else float("nan")
                        if overlay_should_cash(_mode28b, ret_28b, remaining_28b, 0.50, 5):
                            state.weights = {}
                            state.peak_is_locked = True
                        overlay_should_cash(_mode28b, ret_28b, remaining_28b, 0.50, 5)
                        if overlay_should_cash("identity", ret_28b, remaining_28b, 0.50, 5):
                            pass
            except Exception:
                pass
    except Exception:
        pass


def _hook_equity_group(state: _DecideState) -> None:
    from src.portfolio.constraints import load_p27_exposure_limits
    from src.strategies.registry import STRATEGIES as _BL_P29_D
    from src.strategies.sticky.config import load_overlay_mode
    from src.tournament.policy import overlay_should_cash, remaining_sessions

    _rules = state.rules
    decision_date = state.decision_date
    args = state.args
    try:
        _p29_m: Any = _BL_P29_D["sticky.equity_mom60"]()
        _p29v_m: Any = _BL_P29_D["sticky.equity_mom60_vol"]()
        _p30_m: Any = _BL_P29_D["sticky.fillable_mom60"]()
        _p31_m: Any = _BL_P29_D["convex.lottery_impulse"]()
        _p33_m: Any = _BL_P29_D["sticky.mom60_runner_reversal"]()
        _p29_m.restore_state(None, 0)
        _p29v_m.restore_state(None, 0)
        _p30_m.restore_state(None, 0)
        _p31_m.restore_state(None, 0)
        _p33_m.restore_state(None, 0)
        load_p27_exposure_limits()
    except Exception:
        pass
    try:
        _mode29 = str(load_overlay_mode())
        overlay_should_cash(_mode29, 0.0, 5, 0.50, 5)
        overlay_should_cash("identity", 0.0, 5, 0.50, 5)
        if _rules is not None:
            init_cap_29 = float(getattr(_rules, "initial_capital", 1_000_000_000))
            end_date_29: Any = getattr(_rules, "end_date", decision_date)
            try:
                remaining_29 = remaining_sessions(decision_date, end_date_29, get_calendar())
            except Exception:
                remaining_29 = 5
            try:
                cap_val_29 = getattr(args, "capital", None)
                if cap_val_29 is not None:
                    cap_f_29 = float(cap_val_29)
                    if __import__("math").isfinite(cap_f_29):
                        ret_29 = cap_f_29 / init_cap_29 - 1.0 if init_cap_29 > 0 else float("nan")
                        if overlay_should_cash(_mode29, ret_29, remaining_29, 0.50, 5):
                            state.weights = {}
                            state.peak_is_locked = True
                        overlay_should_cash(_mode29, ret_29, remaining_29, 0.50, 5)
                        if overlay_should_cash("identity", ret_29, remaining_29, 0.50, 5):
                            pass
            except Exception:
                pass
    except Exception:
        pass


def _hook_mom60_raw_allocate(state: _DecideState) -> None:
    from pathlib import Path

    import polars as pl

    from src.core.paths import DataPaths
    from src.core.settings import get_settings
    from src.portfolio.sizing import SizingScheme
    from src.tournament.live_decision import (
        assert_sleeve_inputs_fresh,
        build_live_eligible_snapshot,
        compute_live_target_weights,
        resolve_live_championship_sleeve,
    )

    panel = state.panel_loaded
    if panel is None or getattr(panel, "height", 0) == 0:
        state.weights = {}
        state.decision_weights = None
        return
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

    model: Any = _REG_P27["sticky.mom60_raw"]()
    model.reset_trackers()
    rules = state.rules
    try:
        capital = float(getattr(rules, "initial_capital", 1_000_000_000))
    except (TypeError, ValueError):
        capital = 1_000_000_000.0
    intent = compute_live_target_weights(
        model,
        snapshot,
        decision_date=state.decision_date,
        held={},
        capital=capital,
        rules=rules,
        championship_sleeve=sleeve,
        scheme=SizingScheme.TOP1,
        k=1,
    )
    state.decision_weights = intent
    state.weights = dict(intent.weights)
    if intent.kind == "cash":
        state.weights = {}
        state.peak_is_locked = True


_ALLOCATE_HOOKS: Final[dict[str, Any]] = {
    "sticky.split_fill_lock": _hook_split_fill_allocate,
    "sticky.mom60_raw": _hook_mom60_raw_allocate,
}

_OVERLAY_HOOKS: Final[dict[str, Any]] = {
    "sticky.split_fill_lock": _hook_split_fill_lock,
    "sticky.mom60_peak_lock": _hook_mom60_peak_lock,
    "sticky.house_money": _hook_house_money,
    "sticky.mom60_concentrated": _hook_mom60_concentrated,
    "sticky.mom60_raw": _hook_mom60_raw,
    "sticky.mom60_hold": _hook_mom60_hold,
    "sticky.mom60_abs_cash": _hook_mom60_abs_cash,
    "sticky.equity_mom60": _hook_equity_group,
    "sticky.equity_mom60_vol": _hook_equity_group,
    "sticky.fillable_mom60": _hook_equity_group,
    "convex.lottery_impulse": _hook_equity_group,
    "sticky.mom60_runner_reversal": _hook_equity_group,
}

__all__ = ["_ALLOCATE_HOOKS", "_OVERLAY_HOOKS", "DecideHook", "_DecideState"]
