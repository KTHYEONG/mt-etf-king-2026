"""Coverage for the 7 overlay hooks split out of decide/models.py (400-statement budget, R6 AI-context invariant)."""

import argparse
from datetime import date
from types import SimpleNamespace

from src.cli.commands.decide.models import _DecideState, _OVERLAY_HOOKS
from src.cli.commands.decide.models_overlay_hooks import (
    _hook_equity_group,
    _hook_house_money,
    _hook_mom60_abs_cash,
    _hook_mom60_concentrated,
    _hook_mom60_hold,
    _hook_mom60_peak_lock,
    _hook_mom60_raw,
)


def _state(**overrides: object) -> _DecideState:
    defaults: dict[str, object] = {
        "decision_date": date(2026, 9, 10),
        "args": argparse.Namespace(capital=None),
        "model_arg": None,
        "panel_loaded": None,
        "policy": SimpleNamespace(),
        "master": None,
        "rules": SimpleNamespace(initial_capital=1_000_000_000),
        "regime_str": None,
        "lev_allowed": None,
        "inv_allowed": None,
    }
    defaults.update(overrides)
    return _DecideState(**defaults)  # type: ignore[arg-type]


def test_moved_hooks_are_the_exact_objects_registered_in_overlay_hooks() -> None:
    # Given/When: the hook table is built from imports of the split-out module
    # Then: no fork/duplication happened during the split -- same function objects
    assert _OVERLAY_HOOKS["sticky.mom60_peak_lock"] is _hook_mom60_peak_lock
    assert _OVERLAY_HOOKS["sticky.house_money"] is _hook_house_money
    assert _OVERLAY_HOOKS["sticky.mom60_concentrated"] is _hook_mom60_concentrated
    assert _OVERLAY_HOOKS["sticky.mom60_raw"] is _hook_mom60_raw
    assert _OVERLAY_HOOKS["sticky.mom60_hold"] is _hook_mom60_hold
    assert _OVERLAY_HOOKS["sticky.mom60_abs_cash"] is _hook_mom60_abs_cash
    assert _OVERLAY_HOOKS["sticky.equity_mom60"] is _hook_equity_group
    assert _OVERLAY_HOOKS["sticky.fillable_mom60"] is _hook_equity_group
    assert _OVERLAY_HOOKS["convex.lottery_impulse"] is _hook_equity_group


def test_hook_mom60_peak_lock_never_locks_when_capital_equals_initial_capital() -> None:
    # Given: a state whose weights are pre-populated and rules.initial_capital set
    state = _state(weights={"999": 1.0})

    # When: the peak-lock overlay runs (current capital estimate == initial capital in this hook)
    _hook_mom60_peak_lock(state)

    # Then: no drawdown from peak means peak_lock_active is False -- weights untouched
    assert state.peak_is_locked is False
    assert state.weights == {"999": 1.0}


def test_hook_mom60_peak_lock_degrades_safely_when_rules_missing() -> None:
    # Given: no rules object at all
    state = _state(rules=None, weights={"999": 1.0})

    # When: the hook runs (fail-closed try/except idiom, per .agents/rules/quant.md)
    _hook_mom60_peak_lock(state)

    # Then: falls back to the 1_000_000_000 default for both capital and initial_capital -- still no lock
    assert state.peak_is_locked is False
    assert state.weights == {"999": 1.0}
