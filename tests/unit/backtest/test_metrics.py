"""SCENARIO-06-06"""
from __future__ import annotations

import math

from src.backtest.metrics import compound_returns, max_drawdown, window_returns


def test_SCENARIO_06_06_metrics() -> None:
    """SCENARIO-06-06"""
    expected = 1.01**36 - 1
    assert abs(compound_returns([0.01] * 36) - expected) < 1e-12

    win = window_returns([0.01] * 100, 36)
    assert len(win) == 65
    assert all(abs(v - expected) < 1e-12 for v in win)

    assert max_drawdown([100, 120, 90, 150]) == -0.25
