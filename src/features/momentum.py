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
    if frame.height == 0:
        return frame
    if price not in frame.columns:
        return frame
    # Ensure sorted for shift
    sorted_frame = frame.sort([key, "date"])
    result = sorted_frame
    for k in horizons:
        col_name = f"mom_{k}"
        lag = pl.col(price).shift(k).over(key)
        # compute ratio: close / lag -1, guard null or zero
        expr = (
            pl.when(lag.is_null() | (lag == 0) | pl.col(price).is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col(price) / lag - pl.lit(1.0))
            .alias(col_name)
        )
        result = result.with_columns(expr)
    return result
