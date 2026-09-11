from __future__ import annotations

from datetime import date

import polars as pl

from src.features.breadth import cluster_breadth, market_breadth
from tests.unit.features.conftest import session_dates


def _stock_panel(sessions: list[date], above_ma: list[bool]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for i, above in enumerate(above_ma):
        ticker = f"S{i:02d}"
        closes = [90.0 + j for j in range(len(sessions))] if above else [100.0 - j * 0.5 for j in range(len(sessions))]
        for d, close in zip(sessions, closes, strict=True):
            rows.append({"date": d, "ticker": ticker, "close": close})
    return pl.DataFrame(rows)


def test_market_breadth_phantom_session_does_not_null_propagate_ma() -> None:
    """A market-wide phantom session must not null out breadth_ma20-style ratios for the next
    ma_window real trading sessions (rolling_mean with min_samples=w counts physical rows, same
    defect class as add_trend)."""
    sessions = session_dates(date(2026, 1, 2), 15)
    closes = [100.0 + i for i in range(15)]
    closes[10] = None
    panel = pl.DataFrame({"date": sessions, "ticker": ["A"] * 15, "close": closes})
    panel = panel.filter(pl.col("date") <= sessions[11])

    out = market_breadth(panel, sessions[11], ma_window=3, high_window=3)

    assert out.height == 1
    assert out.select("breadth_ma20").item() == 1.0


def test_scenario_05_08_market_and_cluster_breadth() -> None:
    """SCENARIO-05-08"""
    sessions = session_dates(date(2026, 1, 2), 25)
    decision = sessions[-1]
    above = [True] * 6 + [False] * 4
    panel = _stock_panel(sessions, above)
    out = market_breadth(panel, decision, ma_window=20, high_window=20)
    assert out.height == 1
    assert out.select("breadth_ma20").item() == 0.6

    etf_rows: list[dict[str, object]] = []
    for theme, members in [("small", 2), ("big", 3)]:
        for m in range(members):
            ticker = f"{theme}_{m}"
            closes = [100.0] * (len(sessions) - 1) + [120.0]
            for d, close in zip(sessions, closes, strict=True):
                etf_rows.append({"date": d, "ticker": ticker, "theme": theme, "close": close})
    etf_panel = pl.DataFrame(etf_rows)
    cluster = cluster_breadth(etf_panel, "theme", decision, min_members=3, ma_window=20)
    small = cluster.filter(pl.col("theme") == "small").select("breadth_ma20").item()
    big = cluster.filter(pl.col("theme") == "big").select("breadth_ma20").item()
    assert small is None
    assert big == 1.0
