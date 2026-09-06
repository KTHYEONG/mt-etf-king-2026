"""P4 CLI decomposition: context.py carries backtest run context."""
import dataclasses

from src.cli.context import (
    BacktestCellBundle,
    BacktestContext,
    BacktestResult,
    build_backtest_context,
)


def test_p4_context_surface_exists() -> None:
    assert callable(build_backtest_context)
    for cls in (BacktestContext, BacktestCellBundle, BacktestResult):
        assert isinstance(cls, type)
    field_names = {f.name for f in dataclasses.fields(BacktestContext)}
    assert "strategy_id" in field_names
