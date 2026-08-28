from __future__ import annotations

from datetime import date

import polars as pl

from src.features.momentum import add_momentum
from tests.unit.features.conftest import session_dates


def test_scenario_05_03_momentum_compound_and_zero_guard() -> None:
    """SCENARIO-05-03"""
    sessions = session_dates(date(2026, 1, 2), 30)
    closes = [100.0 * (1.01**i) for i in range(30)]
    frame = pl.DataFrame({"date": sessions, "ticker": ["A"] * 30, "close": closes})
    out = add_momentum(frame, [5, 20, 60], date(2026, 12, 31), key="ticker")
    last = out.filter(pl.col("date") == sessions[-1])
    mom20 = last.select("mom_20").item()
    mom5 = last.select("mom_5").item()
    mom60 = last.select("mom_60").item()
    assert mom20 is not None
    assert mom5 is not None
    assert mom60 is None
    assert abs(mom20 - (1.01**20 - 1)) < 1e-12
    assert abs(mom5 - (1.01**5 - 1)) < 1e-12

    zero_frame = pl.DataFrame(
        {
            "date": sessions[:3],
            "ticker": ["Z"] * 3,
            "close": [100.0, 0.0, 101.0],
        }
    )
    zero_out = add_momentum(zero_frame, [1], date(2026, 12, 31), key="ticker")
    assert zero_out.filter(pl.col("date") == sessions[2]).select("mom_1").item() is None
