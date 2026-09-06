"""P4 CLI decomposition: backtest/artifacts.py emission surface."""
from src.cli.commands.backtest.artifacts import (
    attach_backtest_artifacts,
    emit_backtest_artifacts,
)


def test_p4_backtest_artifacts_surface() -> None:
    assert callable(attach_backtest_artifacts)
    assert callable(emit_backtest_artifacts)
