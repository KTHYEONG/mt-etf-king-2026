from __future__ import annotations

from datetime import date

import polars as pl

from src.features.crosssec import add_momentum_acceleration, percentile_rank
from tests.unit.features.conftest import session_dates


def test_scenario_05_06_percentile_rank_and_acceleration() -> None:
    """SCENARIO-05-06"""
    day = session_dates(date(2026, 8, 1), 1)[0]
    frame = pl.DataFrame(
        {
            "date": [day, day, day, day],
            "ticker": ["A", "B", "C", "D"],
            "mom_20": [0.14, 0.08, 0.02, -0.05],
            "mom_5": [0.10, 0.06, 0.01, -0.02],
        }
    )
    ranked = percentile_rank(frame, ["mom_20", "mom_5"], day, by="date", suffix="_rs")
    vals = ranked.sort("mom_20", descending=True).select("mom_20_rs").to_series().to_list()
    assert vals == [1.0, 2 / 3, 1 / 3, 0.0]

    restricted = ranked.filter(pl.col("ticker").is_in(["A", "B"]))
    restricted = percentile_rank(restricted, ["mom_20"], day, by="date", suffix="_rs")
    rs = restricted.sort("ticker").select("mom_20_rs").to_series().to_list()
    assert rs == [1.0, 0.0]

    accel = add_momentum_acceleration(ranked, day, fast=5, slow=20)
    expected = ranked.with_columns((pl.col("mom_5_rs") - pl.col("mom_20_rs")).alias("mom_accel"))
    for ticker in ["A", "B", "C", "D"]:
        got = accel.filter(pl.col("ticker") == ticker).select("mom_accel").item()
        exp = expected.filter(pl.col("ticker") == ticker).select("mom_accel").item()
        assert got == exp
