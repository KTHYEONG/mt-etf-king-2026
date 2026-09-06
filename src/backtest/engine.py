# mypy: ignore-errors
# ruff: noqa
"""Facade preserving the historic `src.backtest.engine` public surface."""
from __future__ import annotations

from src.backtest.execution import Fill  # noqa: F401
from src.backtest.engine_config import (
    BacktestConfig,
    BacktestResult,
    _append_trades_from_transition,
    _build_close_map_ref,
    _policy_ref,
    build_execution_adv,
    logger,
)
from src.backtest.engine_runner import BacktestEngine
from src.backtest.session_grid import resolve_session_grid  # noqa: F401
from src.portfolio.selection import explain_selection_drops  # noqa: F401

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "_append_trades_from_transition",
    "_build_close_map_ref",
    "_policy_ref",
    "build_execution_adv",
    "explain_selection_drops",
    "logger",
    "resolve_session_grid",
]
