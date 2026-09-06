"""Facade fidelity for the src/backtest engine split (P5)."""

from __future__ import annotations


def test_engine_config_canonical_home() -> None:
    import src.backtest.engine as facade
    from src.backtest.engine_config import BacktestConfig, BacktestResult, build_execution_adv

    assert facade.BacktestConfig is BacktestConfig
    assert facade.BacktestResult is BacktestResult
    assert facade.build_execution_adv is build_execution_adv
    assert BacktestConfig.__module__ == "src.backtest.engine_config"
