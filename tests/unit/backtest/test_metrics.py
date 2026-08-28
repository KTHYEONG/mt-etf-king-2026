"""SCENARIO-06-06 SCENARIO-07P-01"""
from __future__ import annotations

import math  # noqa: F401

from src.backtest.metrics import compound_returns, max_drawdown, peak_return, peak_to_final_giveback, terminal_return, window_returns


def test_SCENARIO_06_06_metrics() -> None:  # noqa: N802
    """SCENARIO-06-06"""
    expected = 1.01**36 - 1
    assert abs(compound_returns([0.01] * 36) - expected) < 1e-12

    win = window_returns([0.01] * 100, 36)
    assert len(win) == 65
    assert all(abs(v - expected) < 1e-12 for v in win)

    assert max_drawdown([100, 120, 90, 150]) == -0.25


def test_SCENARIO_07P_01_peak_giveback() -> None:  # noqa: N802
    """SCENARIO-07P-01"""
    assert abs(peak_return([100, 150, 120]) - 0.5) < 1e-12
    assert abs(terminal_return([100, 150, 120]) - 0.2) < 1e-12
    assert abs(peak_to_final_giveback([100, 150, 120]) - 0.3) < 1e-12
    assert abs(peak_to_final_giveback([1.0, 1.1, 1.21])) < 1e-12
    assert peak_return([]) == 0.0
    assert terminal_return([]) == 0.0
    assert peak_to_final_giveback([]) == 0.0
    assert peak_return([0.0, 1.0]) == 0.0
    assert terminal_return([0.0, 1.0]) == 0.0
    assert peak_to_final_giveback([0.0, 1.0]) == 0.0
