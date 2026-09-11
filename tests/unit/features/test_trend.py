from __future__ import annotations

from datetime import date

import polars as pl

import pytest

from src.features.trend import add_trend
from tests.unit.features.conftest import session_dates


def test_add_trend_phantom_session_does_not_null_propagate_for_window_sessions() -> None:
    """A single market-wide phantom session (no trading) must not null out ma_w/roll_max_w/
    drawdown_w for the next w real trading sessions across the whole panel: rolling_mean/max with
    min_samples=w count PHYSICAL rows, so without excluding the phantom row first, every window
    still containing it is null even though w real trading sessions are actually available."""
    sessions = session_dates(date(2026, 1, 2), 15)
    closes = [100.0 + i for i in range(15)]
    closes[10] = None  # market-wide phantom session (no trades that day)
    frame = pl.DataFrame({"date": sessions, "ticker": ["A"] * 15, "close": closes})

    out = add_trend(frame, [3], [3], date(2026, 12, 31), key="ticker")

    # The phantom session itself has no price -> no valid ma_3 that day.
    assert out.filter(pl.col("date") == sessions[10]).select("ma_3").item() is None
    # First real session after the phantom: OLD (buggy) code would still see the null
    # physically inside its trailing 3-row window and emit None. Skipping the phantom
    # session, the trailing 3 REAL closes are indices 8, 9, 11 -> (108+109+111)/3.
    ma3_next = out.filter(pl.col("date") == sessions[11]).select("ma_3").item()
    assert ma3_next == pytest.approx((closes[8] + closes[9] + closes[11]) / 3)
    roll_max_next = out.filter(pl.col("date") == sessions[11]).select("roll_max_3").item()
    assert roll_max_next == pytest.approx(max(closes[8], closes[9], closes[11]))
    dd_next = out.filter(pl.col("date") == sessions[11]).select("drawdown_3").item()
    assert dd_next is not None
    assert dd_next == pytest.approx((closes[11] - roll_max_next) / roll_max_next)


def test_add_trend_emits_ma_columns() -> None:
    sessions = session_dates(date(2026, 1, 2), 30)
    frame = pl.DataFrame(
        {
            "date": sessions,
            "ticker": ["A"] * len(sessions),
            "close": [100.0 + i for i in range(len(sessions))],
        }
    )
    out = add_trend(frame, [20], [20], date(2026, 12, 31), key="ticker")
    assert "ma_20" in out.columns
    assert "ma_ratio_20" in out.columns
    assert "breakout_20" in out.columns


def test_add_trend_no_windows_returns_frame_unchanged() -> None:
    sessions = session_dates(date(2026, 1, 2), 5)
    frame = pl.DataFrame({"date": sessions, "ticker": ["A"] * 5, "close": [100.0 + i for i in range(5)]})
    out = add_trend(frame, [], [], date(2026, 12, 31), key="ticker")
    assert out.equals(frame.sort(["ticker", "date"]))
