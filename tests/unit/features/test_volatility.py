from __future__ import annotations

from datetime import date

import polars as pl

from src.features.volatility import add_volatility
from tests.unit.features.conftest import session_dates


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
