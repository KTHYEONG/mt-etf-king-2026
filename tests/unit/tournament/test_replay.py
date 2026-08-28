"""SCENARIO-06-11"""
from __future__ import annotations

from datetime import date

import polars as pl

from src.alpha.baselines import BASELINES
from src.backtest.costs import CostConfig
from src.backtest.engine import BacktestConfig
from src.backtest.metrics import compound_returns
from src.portfolio.sizing import SizingScheme
from src.tournament.replay import TournamentReplay
from tests.unit.backtest.conftest import build_engine, panel_row


def test_SCENARIO_06_11_tournament_replay_2025() -> None:
    """SCENARIO-06-11"""
    start = date(2025, 9, 22)
    end = date(2025, 11, 14)
    cal = __import__("src.core.calendar", fromlist=["TradingCalendar"]).TradingCalendar()
    sessions = cal.sessions(start, end)
    assert cal.session_count(start, end) == 35
    rows: list[dict[str, object]] = []
    for i, d in enumerate(sessions):
        rows.append(
            panel_row(
                day=d,
                ticker="069500",
                close=30_000.0 + i * 10,
                open_=29_900.0 + i * 10,
                mom_20=0.05 + i * 0.001,
                name="KODEX 200",
                theme="반도체",
            )
        )
        rows.append(
            panel_row(
                day=d,
                ticker="451060",
                close=20_000.0 + i * 5,
                open_=19_900.0 + i * 5,
                mom_20=0.03 + i * 0.001,
                name="KODEX K-반도체",
                theme="반도체",
            )
        )
    panel = pl.DataFrame(rows)
    engine, cal, filt = build_engine(panel)
    config = BacktestConfig(
        start=start,
        end=end,
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    report = TournamentReplay(engine, cal).run(BASELINES["B1"](), panel, config)
    assert report.sessions == 35
    assert len(report.days) == 35
    assert report.days[-1].execution_date is None
    daily = [day.daily_return for day in report.days]
    assert abs(report.days[-1].cumulative_return - compound_returns(daily)) < 1e-10
    for day in report.days:
        assert day.regime
        assert isinstance(day.dropped, dict)
        if day.top_scores:
            assert day.top_scores[0][1]
