"""Cost × participation stress harness."""
from __future__ import annotations

from src.backtest.costs import CostConfig
from src.tournament.harness import harness_case_count, iter_harness_cases, load_participation_grid


def test_load_participation_grid_from_config() -> None:
    grid = load_participation_grid()
    assert grid == (0.01, 0.02, 0.05)


def test_iter_harness_cases_cost_and_participation_grid() -> None:
    cases = list(iter_harness_cases(CostConfig(), participation_grid=(0.01, 0.02)))
    assert len(cases) == len(CostConfig().grid()) * 2
    assert all(c.commission_bps is not None for c, _ in cases)
    assert all(c.slippage_bps is not None for c, _ in cases)
    participations = {p for _, p in cases}
    assert participations == {0.01, 0.02}


def test_harness_case_count() -> None:
    assert harness_case_count(CostConfig(0.0, 0.0, 0.0), participation_grid=(0.01, 0.05)) == 2
