"""Facade fidelity for the src backtest engine runner split (P5)."""

from __future__ import annotations


def test_engine_runner_canonical_home() -> None:
    import src.backtest.engine as facade
    from src.backtest.engine_runner import BacktestEngine

    assert facade.BacktestEngine is BacktestEngine
    assert BacktestEngine.__module__ == "src.backtest.engine_runner"
