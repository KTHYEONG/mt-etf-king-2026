from __future__ import annotations

from datetime import date

import polars as pl

from src.features.trend import add_trend
from tests.unit.features.conftest import session_dates


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
