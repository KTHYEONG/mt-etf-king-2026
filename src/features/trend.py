from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from src.features.pit import assert_pit


def add_trend(
    frame: pl.DataFrame,
    ma_windows: Sequence[int],
    breakout_windows: Sequence[int],
    decision_date: date,
    key: str = "ticker",
) -> pl.DataFrame:
    assert_pit(frame, decision_date)
    if frame.height == 0:
        return frame
    if "close" not in frame.columns:
        return frame
    sorted_frame = frame.sort([key, "date"])
    result = sorted_frame
    # MA and ratio / slope
    for w in ma_windows:
        ma_col = f"ma_{w}"
        ratio_col = f"ma_ratio_{w}"
        slope_col = f"ma_slope_{w}"
        # rolling mean on session-aligned grid
        ma_expr = pl.col("close").rolling_mean(window_size=w, min_samples=w).over(key).alias(ma_col)
        result = result.with_columns(ma_expr)
        # ratio close / ma
        result = result.with_columns(
            pl.when(pl.col(ma_col).is_null() | (pl.col(ma_col) == 0) | pl.col("close").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("close") / pl.col(ma_col))
            .alias(ratio_col)
        )
        # slope: ma / ma.shift(1) -1
        ma_lag = pl.col(ma_col).shift(1).over(key)
        result = result.with_columns(
            pl.when(ma_lag.is_null() | (ma_lag == 0) | pl.col(ma_col).is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col(ma_col) / ma_lag - pl.lit(1.0))
            .alias(slope_col)
        )
    # Breakout: close vs rolling max, and drawdown
    for w in breakout_windows:
        high_col = f"roll_max_{w}"
        breakout_col = f"breakout_{w}"
        dd_col = f"drawdown_{w}"
        max_expr = pl.col("close").rolling_max(window_size=w, min_samples=w).over(key).alias(high_col)
        result = result.with_columns(max_expr)
        # breakout = close / roll_max -1 ? or binary? Use ratio -1
        result = result.with_columns(
            pl.when(pl.col(high_col).is_null() | (pl.col(high_col) == 0) | pl.col("close").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("close") / pl.col(high_col) - pl.lit(1.0))
            .alias(breakout_col)
        )
        result = result.with_columns(
            pl.when(pl.col(high_col).is_null() | (pl.col(high_col) == 0) | pl.col("close").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise((pl.col("close") - pl.col(high_col)) / pl.col(high_col))
            .alias(dd_col)
        )
    return result
