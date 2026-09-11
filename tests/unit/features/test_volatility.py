from __future__ import annotations

import math
from datetime import date

import polars as pl

from src.features.volatility import add_volatility
from tests.unit.features.conftest import session_dates


def test_add_volatility_phantom_session_does_not_null_propagate_rv() -> None:
    """A market-wide phantom session must not null out rv_w for the next w real trading
    sessions (rolling_std with min_samples=w counts physical rows, same defect class as
    add_trend/roll_max)."""
    sessions = session_dates(date(2026, 1, 2), 15)
    closes = [100.0 + i for i in range(15)]
    closes[10] = None
    frame = pl.DataFrame(
        {
            "date": sessions,
            "ticker": ["A"] * 15,
            "close": closes,
            "open": closes,
            "high": closes,
            "low": closes,
        }
    )

    out = add_volatility(frame, [3], date(2026, 12, 31), key="ticker")

    rv3_next = out.filter(pl.col("date") == sessions[11]).select("rv_3").item()
    assert rv3_next is not None
    assert math.isfinite(rv3_next)


def test_add_volatility_emits_rv_columns() -> None:
    sessions = session_dates(date(2026, 1, 2), 30)
    frame = pl.DataFrame(
        {
            "date": sessions,
            "ticker": ["A"] * len(sessions),
            "close": [100.0 + i * 0.5 for i in range(len(sessions))],
            "open": [99.0 + i * 0.5 for i in range(len(sessions))],
            "high": [101.0 + i * 0.5 for i in range(len(sessions))],
            "low": [98.0 + i * 0.5 for i in range(len(sessions))],
        }
    )
    out = add_volatility(frame, [5, 20], date(2026, 12, 31), key="ticker")
    assert "rv_5" in out.columns
    assert "rv_20" in out.columns


def test_add_volatility_no_windows_returns_frame_unchanged() -> None:
    sessions = session_dates(date(2026, 1, 2), 5)
    frame = pl.DataFrame({"date": sessions, "ticker": ["A"] * 5, "close": [100.0 + i for i in range(5)]})
    out = add_volatility(frame, [], date(2026, 12, 31), key="ticker")
    assert out.equals(frame.sort(["ticker", "date"]))
