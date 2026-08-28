"""SCENARIO-06-05"""
from __future__ import annotations

from src.backtest.costs import CostConfig, CostModel


def test_SCENARIO_06_05_cost_grid_and_charge() -> None:
    """SCENARIO-06-05"""
    grid = CostConfig(commission_bps=None, slippage_bps=None, spread_bps=0.0).grid()
    assert len(grid) == 12
    assert all(c.commission_bps is not None for c in grid)
    assert all(c.slippage_bps is not None for c in grid)
    assert all(c.spread_bps is not None for c in grid)

    single = CostConfig(0.0, 0.0, 0.0).grid()
    assert len(single) == 1

    model = CostModel(CostConfig(3.0, 5.0, 0.0))
    assert model.charge(1_000_000_000) == 800_000.0
