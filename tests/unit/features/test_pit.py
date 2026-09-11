from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.features.momentum import add_momentum
from src.features.pit import PitViolationError, align_session_grid, assert_pit, restrict_to_traded_sessions
from tests.unit.features.conftest import session_dates


def test_scenario_05_01_assert_pit_violation_and_pass() -> None:
    """SCENARIO-05-01"""
    frame = pl.DataFrame(
        {
            "date": [date(2026, 8, 26), date(2026, 8, 27), date(2026, 8, 28)],
            "ticker": ["A", "A", "A"],
            "close": [100.0, 101.0, 102.0],
        }
    )
    with pytest.raises(PitViolationError, match="2026-08-28"):
        assert_pit(frame, date(2026, 8, 27))
    out = assert_pit(frame, date(2026, 8, 28))
    assert out.equals(frame)


def test_scenario_05_02_align_session_grid_momentum_lag() -> None:
    """SCENARIO-05-02"""
    sessions = session_dates(date(2026, 1, 2), 10)
    rows_a = [{"date": d, "ticker": "A", "close": 100.0 + i} for i, d in enumerate(sessions)]
    rows_b = [
        {"date": d, "ticker": "B", "close": 200.0 + i}
        for i, d in enumerate(sessions)
        if i not in (3, 4)
    ]
    raw = pl.DataFrame(rows_a + rows_b)
    aligned = align_session_grid(raw, sessions, key="ticker")
    assert aligned.filter(pl.col("ticker") == "A").height == 10
    assert aligned.filter(pl.col("ticker") == "B").height == 10
    b_nulls = aligned.filter((pl.col("ticker") == "B") & pl.col("close").is_null()).select("date").to_series().to_list()
    assert sessions[3] in b_nulls
    assert sessions[4] in b_nulls
    mom = add_momentum(aligned, [5], date(2026, 12, 31), key="ticker")
    b = mom.filter(pl.col("ticker") == "B").sort("date")
    s8 = sessions[8]
    s9 = sessions[9]
    assert b.filter(pl.col("date") == s8).select("mom_5").item() is None
    assert b.filter(pl.col("date") == s9).select("mom_5").item() is None


def test_restrict_to_traded_sessions_drops_only_market_wide_phantom_dates() -> None:
    sessions = session_dates(date(2026, 1, 2), 6)
    frame = pl.DataFrame(
        {
            "date": sessions + sessions,
            "ticker": ["A"] * 6 + ["B"] * 6,
            # index 2 is a market-wide phantom (both A and B null); index 4 is a
            # single-ticker gap (A only) and must be KEPT since B still traded that date.
            "close": [100.0, 101.0, None, 103.0, None, 105.0, 200.0, 201.0, None, 203.0, 204.0, 205.0],
        }
    )
    out = restrict_to_traded_sessions(frame, price_col="close")
    assert out.height == 10
    assert sessions[2] not in out["date"].to_list()
    assert sessions[4] in out["date"].to_list()
    assert out.filter((pl.col("ticker") == "A") & (pl.col("date") == sessions[4])).height == 1


def test_restrict_to_traded_sessions_fail_closed_inputs() -> None:
    empty = pl.DataFrame({"date": [], "close": []})
    assert restrict_to_traded_sessions(empty, price_col="close").height == 0
    no_price_col = pl.DataFrame({"date": [date(2026, 1, 2)], "ticker": ["A"]})
    assert restrict_to_traded_sessions(no_price_col, price_col="close").equals(no_price_col)
    no_date_col = pl.DataFrame({"close": [100.0]})
    assert restrict_to_traded_sessions(no_date_col, price_col="close").equals(no_date_col)
