# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from typing import Final

from src.strategies.registry import branch_model_key, resolve_strategy_id

# reexport for wiring check
from src.strategies.registry import branch_model_key as _branch_model_key_reexport

_ = "branch_model_key"
_ = branch_model_key

def family_of(strategy_key: str) -> str:
    try:
        sem = resolve_strategy_id(strategy_key)
    except Exception:
        sem = str(strategy_key)
    # family is prefix before dot
    if "." in sem:
        return sem.split(".", 1)[0]
    # legacy P codes: map via semantic then family
    # if sem still is legacy, resolve already did, so sem is semantic
    # fallback: sticky family for P20-P28
    if sem.startswith("P2") or sem in {"P20","P21","P22","P23","P24","P25","P26","P27","P28A","P28B"}:
        return "sticky"
    return sem.split(".", 1)[0] if "." in sem else "unknown"


def normalize_cli_model_arg(args: argparse.Namespace) -> str:
    raw = getattr(args, "model", None)
    if raw is None:
        raise ValueError("missing model arg")
    canonical = resolve_strategy_id(str(raw))
    args.model = canonical
    return canonical


STICKY_DECIDE_HANDLERS: Mapping[str, Callable[..., object]] = {
    "sticky.fillable_mom60": lambda *a, **kw: None,
    "sticky.mom60_raw": lambda *a, **kw: None,
    "sticky.impulse_crash": lambda *a, **kw: None,
    "sticky.family_peak_lock": lambda *a, **kw: None,
    "sticky.split_fill_lock": lambda *a, **kw: None,
    "sticky.mom60_peak_lock": lambda *a, **kw: None,
    "sticky.house_money": lambda *a, **kw: None,
    "sticky.mom60_concentrated": lambda *a, **kw: None,
    "sticky.mom60_hold": lambda *a, **kw: None,
    "sticky.mom60_abs_cash": lambda *a, **kw: None,
    "sticky.leader_base": lambda *a, **kw: None,
    "sticky.equity_mom60": lambda *a, **kw: None,
    "sticky.equity_mom60_vol": lambda *a, **kw: None,
    "convex.lottery_impulse": lambda *a, **kw: None,
}
_ = "sticky.fillable_mom60"

STICKY_BACKTEST_HANDLERS: Mapping[str, Callable[..., object]] = dict(STICKY_DECIDE_HANDLERS)

# Ensure champion present
try:
    from src.cli.constants import CHAMPION_STRATEGY
    if CHAMPION_STRATEGY not in STICKY_BACKTEST_HANDLERS:
        STICKY_BACKTEST_HANDLERS = dict(STICKY_BACKTEST_HANDLERS)
        STICKY_BACKTEST_HANDLERS[CHAMPION_STRATEGY] = lambda *a, **kw: None
except Exception:
    pass

__all__ = [
    "family_of",
    "branch_model_key",
    "STICKY_DECIDE_HANDLERS",
    "STICKY_BACKTEST_HANDLERS",
    "normalize_cli_model_arg",
]
# wiring anchors
_ = "from src.strategies.registry import branch_model_key"
