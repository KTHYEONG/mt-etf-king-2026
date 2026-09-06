# ruff: noqa
# mypy: ignore-errors
"""Facade preserving the historic `src.tournament.simulator` public surface."""
from __future__ import annotations

from src.backtest.session_grid import resolve_session_grid
from src.tournament.simulator_rolling import RollingDiagnostics, RollingResult, TournamentSimulator
from src.tournament.simulator_windows import (
    _path_dependent_ref,
    model_requires_path_dependent,
    oneshot_independent_window_returns,
    resolve_prestart_intent,
    simulate_window_from_cache,
)

__all__ = [
    "RollingDiagnostics",
    "RollingResult",
    "TournamentSimulator",
    "_path_dependent_ref",
    "model_requires_path_dependent",
    "oneshot_independent_window_returns",
    "resolve_prestart_intent",
    "resolve_session_grid",
    "simulate_window_from_cache",
]
