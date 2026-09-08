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
            "date": sessions[:3] + sessions[:3],
            "ticker": ["Z"] * 3 + ["Y"] * 3,
            "close": [100.0, 0.0, 101.0, 200.0, 210.0, 220.0],
        }
    )
    zero_out = add_momentum(zero_frame, [1], date(2026, 12, 31), key="ticker")
    assert zero_out.filter((pl.col("date") == sessions[2]) & (pl.col("ticker") == "Z")).select("mom_1").item() is None


def test_add_momentum_skips_globally_invalid_phantom_session() -> None:
    from datetime import date

    import polars as pl
    import pytest

    from src.features.momentum import add_momentum

    days = [date(2026, 6, d) for d in (1, 2, 3, 4, 5)]
    frame = pl.DataFrame(
        {
            "date": days + days,
            "ticker": ["A"] * 5 + ["B"] * 5,
            "close": [100.0, 110.0, None, 121.0, 133.1, 200.0, 220.0, None, 242.0, 266.2],
        }
    )
    out = add_momentum(frame, [2], days[-1])
    assert out.height == frame.height
    assert out.filter((pl.col("date") == days[2]) & (pl.col("ticker") == "A"))["mom_2"][0] is None
    assert out.filter((pl.col("date") == days[3]) & (pl.col("ticker") == "A"))["mom_2"][0] == pytest.approx(0.21)
    assert out.filter((pl.col("date") == days[4]) & (pl.col("ticker") == "B"))["mom_2"][0] == pytest.approx(0.21)


def test_add_momentum_keeps_ticker_missing_on_valid_session_fail_closed() -> None:
    from datetime import date

    import polars as pl
    import pytest

    from src.features.momentum import add_momentum

    days = [date(2026, 7, d) for d in (1, 2, 3)]
    frame = pl.DataFrame(
        {
            "date": days + days,
            "ticker": ["A"] * 3 + ["B"] * 3,
            "close": [100.0, None, 121.0, 200.0, 210.0, 220.0],
        }
    )
    out = add_momentum(frame, [1], days[-1])
    assert out.filter((pl.col("date") == days[2]) & (pl.col("ticker") == "A"))["mom_1"][0] is None
    assert out.filter((pl.col("date") == days[2]) & (pl.col("ticker") == "B"))["mom_1"][0] == pytest.approx(220.0 / 210.0 - 1.0)
    for bad in ([0], [-1], [True], [1.5], [1, 1]):
        with pytest.raises(ValueError, match="horizon"):
            add_momentum(frame, bad, days[-1])
