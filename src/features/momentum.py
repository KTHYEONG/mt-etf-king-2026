from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from src.features.pit import assert_pit


def add_momentum(
    frame: pl.DataFrame,
    horizons: Sequence[int],
    decision_date: date,
    price: str = "close",
    key: str = "ticker",
) -> pl.DataFrame:
    assert_pit(frame, decision_date)
    horizons = list(horizons)
    seen: set[int] = set()
    for h in horizons:
        if isinstance(h, bool) or not isinstance(h, int) or h <= 0 or h in seen:
            raise ValueError(f"invalid horizon {h!r}: horizons must be positive unique integers")
        seen.add(h)
    if frame.height == 0:
        return frame
    if price not in frame.columns or key not in frame.columns or "date" not in frame.columns:
        return frame
    sorted_frame = frame.sort([key, "date"])
    valid_flag = (
        pl.col(price).is_not_null() & pl.col(price).is_finite() & (pl.col(price) > 0.0)
    ).any().over("date")
    valid_dates = sorted_frame.filter(valid_flag).select(pl.col("date").unique()).to_series().to_list()
    calc = sorted_frame.filter(pl.col("date").is_in(valid_dates))
    lag_exprs: list[pl.Expr] = []
    for h in horizons:
        col_name = f"mom_{h}"
        lag = pl.col(price).shift(h).over(key)
        cur = pl.col(price)
        invalid = (
            cur.is_null() | cur.is_not_null() & (~cur.is_finite() | (cur <= 0.0)) | lag.is_null() | lag.is_not_null() & (~lag.is_finite() | (lag <= 0.0))
        )
        lag_exprs.append(
            pl.when(invalid)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise((cur.cast(pl.Float64) / lag.cast(pl.Float64) - pl.lit(1.0, dtype=pl.Float64)).cast(pl.Float64))
            .alias(col_name)
        )
    calc = calc.with_columns(lag_exprs) if lag_exprs else calc
    select_cols = [key, "date"] + [f"mom_{h}" for h in horizons]
    calc_select = calc.select(select_cols)
    result = sorted_frame.join(calc_select, on=[key, "date"], how="left")
    return result
