"""SCENARIO-11-01 SCENARIO-11-02 SCENARIO-11-05 SCENARIO-11-06"""
from __future__ import annotations

import inspect
import time
from datetime import date

import polars as pl
import pytest

from src.backtest.costs import CostConfig
from src.backtest.engine import BacktestConfig
from src.backtest.session_cache import build_session_cache
from src.core.calendar import TradingCalendar
from src.portfolio.policy import PortfolioPolicy
from src.portfolio.sizing import ConfidenceSizingConfig, SizingScheme
from src.tournament.harness import (
    DEFAULT_PROTOCOL_COMMISSION_BPS,
    DEFAULT_PROTOCOL_PARTICIPATION,
    DEFAULT_PROTOCOL_SLIPPAGE_BPS,
    harness_case_count,
    iter_harness_cases,
    iter_protocol_cases,
)
from src.tournament.simulator import TournamentSimulator
from tests.unit.backtest.conftest import build_engine, panel_row


def _policy_with_score() -> PortfolioPolicy:
    policy = PortfolioPolicy(sizing_config=ConfidenceSizingConfig())
    policy.name = "P08"  # type: ignore[attr-defined]
    policy.scores_path_independent = True  # type: ignore[attr-defined]

    def _score(snapshot, ctx):
        scores: dict[str, float] = {}
        for row in snapshot.iter_rows(named=True):
            t = str(row.get("ticker"))
            v = row.get("mom_20")
            try:
                scores[t] = float(v) if v is not None else 0.0
            except Exception:  # noqa: S112
                continue
        if not scores:
            scores = {"069500": 1.0}
        return scores

    policy.score = _score  # type: ignore[attr-defined]
    return policy


def _synthetic_panel(sessions: list[date]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for i, d in enumerate(sessions):
        rows.append(panel_row(day=d, ticker="069500", close=30000.0 + i * 10, mom_20=0.20 + i * 0.001))
        rows.append(panel_row(day=d, ticker="451060", close=20000.0 + i * 10, mom_20=0.10 + i * 0.001))
    return pl.DataFrame(rows)


def test_SCENARIO_11_01_iter_protocol_cases_single() -> None:  # noqa: N802
    """SCENARIO-11-01"""
    cases = list(iter_protocol_cases("single"))
    assert len(cases) == 1
    cost, participation = cases[0]
    assert cost.commission_bps == DEFAULT_PROTOCOL_COMMISSION_BPS == 3.0
    assert cost.slippage_bps == DEFAULT_PROTOCOL_SLIPPAGE_BPS == 5.0
    assert cost.spread_bps == 0.0
    assert participation == DEFAULT_PROTOCOL_PARTICIPATION == 0.01


def test_SCENARIO_11_02_iter_protocol_cases_grid() -> None:  # noqa: N802
    """SCENARIO-11-02"""
    grid_cases = list(iter_protocol_cases("grid"))
    harness_cases = list(iter_harness_cases(CostConfig()))
    assert len(grid_cases) == harness_case_count(CostConfig()) == 36
    assert grid_cases == harness_cases


def test_SCENARIO_11_05_shared_session_cache_equivalence() -> None:  # noqa: N802
    """SCENARIO-11-05"""
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 2, 28))
    panel = _synthetic_panel(sessions)
    engine, cal2, filt = build_engine(panel)
    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    policy = _policy_with_score()
    sim = TournamentSimulator(engine, cal2)
    without_cache = sim.run_rolling(
        policy,
        panel,
        config,
        horizon=5,
        path_dependent=True,
        path_dependent_mode="fast",
        session_cache=None,
    )
    cache = build_session_cache(engine, policy, panel, config)
    with_cache = sim.run_rolling(
        policy,
        panel,
        config,
        horizon=5,
        path_dependent=True,
        path_dependent_mode="fast",
        session_cache=cache,
    )
    assert len(without_cache.returns) == len(with_cache.returns)
    max_diff = max(abs(a - b) for a, b in zip(without_cache.returns, with_cache.returns, strict=True))
    assert max_diff <= 1e-12, f"return diff {max_diff}"


def test_SCENARIO_11_06_fast_default_speedup() -> None:  # noqa: N802
    """SCENARIO-11-06"""
    sig = inspect.signature(TournamentSimulator.run_rolling)
    assert sig.parameters["path_dependent_mode"].default == "fast"

    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 2, 28))
    panel = _synthetic_panel(sessions)
    engine, cal2, filt = build_engine(panel)
    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    policy = _policy_with_score()
    sim = TournamentSimulator(engine, cal2)
    t0 = time.perf_counter()
    slow = sim.run_rolling(policy, panel, config, horizon=5, path_dependent=True, path_dependent_mode="slow")
    t1 = time.perf_counter()
    fast = sim.run_rolling(policy, panel, config, horizon=5, path_dependent=True)
    t2 = time.perf_counter()
    wall_slow = t1 - t0
    wall_fast = t2 - t1
    assert len(slow.returns) == len(fast.returns)
    assert wall_fast * 3 < wall_slow, f"speedup {wall_slow / wall_fast if wall_fast else 0:.1f}x <3"


@pytest.mark.parametrize(
    "scenario_id",
    ["SCENARIO-11-01", "SCENARIO-11-02", "SCENARIO-11-05", "SCENARIO-11-06"],
)
def test_SCENARIO_hyphen_wrapper_perf_cut(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-11-01":
        test_SCENARIO_11_01_iter_protocol_cases_single()
    elif scenario_id == "SCENARIO-11-02":
        test_SCENARIO_11_02_iter_protocol_cases_grid()
    elif scenario_id == "SCENARIO-11-05":
        test_SCENARIO_11_05_shared_session_cache_equivalence()
    elif scenario_id == "SCENARIO-11-06":
        test_SCENARIO_11_06_fast_default_speedup()
