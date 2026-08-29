"""SCENARIO-PERF-01 SCENARIO-PERF-03 SCENARIO-PERF-04"""
from __future__ import annotations

import time
from datetime import date
from unittest.mock import MagicMock

import polars as pl
import pytest

from src.backtest.costs import CostConfig
from src.backtest.engine import BacktestConfig
from src.core.calendar import TradingCalendar
from src.portfolio.policy import PortfolioPolicy
from src.portfolio.sizing import ConfidenceSizingConfig, SizingScheme
from src.tournament.simulator import TournamentSimulator
from tests.unit.backtest.conftest import build_engine, panel_row


def _policy_with_score():
    policy = PortfolioPolicy(sizing_config=ConfidenceSizingConfig())
    policy.name = "P08"  # type: ignore[attr-defined]

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


def test_SCENARIO_PERF_01_fast_vs_slow_equivalence() -> None:  # noqa: N802
    """SCENARIO-PERF-01"""
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 2, 28))
    panel = _synthetic_panel(sessions)
    engine, cal2, filt = build_engine(panel)
    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig(0.0, 0.0, 0.0))
    policy = _policy_with_score()
    sim = TournamentSimulator(engine, cal2)
    slow = sim.run_rolling(policy, panel, config, horizon=5, path_dependent=True, path_dependent_mode="slow")
    fast = sim.run_rolling(policy, panel, config, horizon=5, path_dependent=True, path_dependent_mode="fast")
    assert len(fast.returns) == len(slow.returns)
    assert len(fast.returns) > 0
    max_ret = max(abs(a - b) for a, b in zip(fast.returns, slow.returns, strict=True))
    max_dd = max(abs(a - b) for a, b in zip(fast.drawdowns, slow.drawdowns, strict=True))
    max_gb = max(abs(a - b) for a, b in zip(fast.givebacks, slow.givebacks, strict=True))
    assert max_ret <= 1e-12, f"ret diff {max_ret}"
    assert max_dd <= 1e-12, f"dd diff {max_dd}"
    assert max_gb <= 1e-12, f"gb diff {max_gb}"


def test_SCENARIO_PERF_03_engine_run_counts() -> None:  # noqa: N802
    """SCENARIO-PERF-03"""
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 2, 28))
    panel = _synthetic_panel(sessions)
    engine, cal2, filt = build_engine(panel)
    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig(0.0, 0.0, 0.0))
    policy = _policy_with_score()
    # slow must call engine.run exactly n_windows times
    engine_slow = build_engine(panel)[0]
    sim_slow = TournamentSimulator(engine_slow, cal2)
    mock_slow = MagicMock(wraps=engine_slow.run)
    engine_slow.run = mock_slow  # type: ignore[method-assign]
    res_slow = sim_slow.run_rolling(policy, panel, config, horizon=5, path_dependent=True, path_dependent_mode="slow")
    assert mock_slow.call_count == len(res_slow.returns)
    # fast must not call engine.run in per-window loop
    engine_fast, _, _ = build_engine(panel)
    sim_fast = TournamentSimulator(engine_fast, cal2)
    mock_fast = MagicMock(wraps=engine_fast.run)
    engine_fast.run = mock_fast  # type: ignore[method-assign]
    res_fast = sim_fast.run_rolling(policy, panel, config, horizon=5, path_dependent=True, path_dependent_mode="fast")
    assert mock_fast.call_count == 0, f"fast called {mock_fast.call_count}"
    assert len(res_fast.returns) == len(res_slow.returns)


def test_SCENARIO_PERF_04_speedup() -> None:  # noqa: N802
    """SCENARIO-PERF-04"""
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 6, 30))
    assert len(sessions) >= 80
    panel = _synthetic_panel(sessions)
    engine, cal2, filt = build_engine(panel)
    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig(0.0, 0.0, 0.0))
    policy = _policy_with_score()
    sim = TournamentSimulator(engine, cal2)
    t0 = time.perf_counter()
    slow = sim.run_rolling(policy, panel, config, horizon=10, path_dependent=True, path_dependent_mode="slow")
    t1 = time.perf_counter()
    fast = sim.run_rolling(policy, panel, config, horizon=10, path_dependent=True, path_dependent_mode="fast")
    t2 = time.perf_counter()
    wall_slow = t1 - t0
    wall_fast = t2 - t1
    assert len(slow.returns) == len(fast.returns)
    assert wall_fast * 10 <= wall_slow, f"speedup {wall_slow/wall_fast if wall_fast else 0} <10 wall_fast={wall_fast} wall_slow={wall_slow}"


@pytest.mark.parametrize("scenario_id", ["SCENARIO-PERF-01", "SCENARIO-PERF-03", "SCENARIO-PERF-04"])
def test_SCENARIO_PERF_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-PERF-01":
        test_SCENARIO_PERF_01_fast_vs_slow_equivalence()
    elif scenario_id == "SCENARIO-PERF-03":
        test_SCENARIO_PERF_03_engine_run_counts()
    elif scenario_id == "SCENARIO-PERF-04":
        test_SCENARIO_PERF_04_speedup()
