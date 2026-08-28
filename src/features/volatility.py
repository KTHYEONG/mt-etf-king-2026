from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from src.features.pit import assert_pit


def add_volatility(
    frame: pl.DataFrame,
    windows: Sequence[int],
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
    # compute daily simple return for volatility (using close/prev -1)
    # We create a return column for internal rolling std, but not expose?
    # Use over group
    # Create log return concept? Use simple return for std.
    ret_expr = (
        pl.when(pl.col("close").shift(1).over(key).is_null() | (pl.col("close").shift(1).over(key) == 0))
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("close") / pl.col("close").shift(1).over(key) - pl.lit(1.0))
        .alias("_ret")
    )
    result = result.with_columns(ret_expr)
    for w in windows:
        rv_col = f"rv_{w}"
        # rolling std of return
        rv_expr = pl.col("_ret").rolling_std(window_size=w, min_samples=w).over(key).alias(rv_col)
        result = result.with_columns(rv_expr)
        # ATR-like: (high-low)/close rolling mean
        if "high" in frame.columns and "low" in frame.columns:
            tr_col = f"_tr_{w}"
            # true range approximation: high - low divided by close
            result = result.with_columns(
                pl.when(pl.col("close").is_null() | (pl.col("close") == 0))
                .then(pl.lit(None, dtype=pl.Float64))
                .otherwise((pl.col("high") - pl.col("low")) / pl.col("close"))
                .alias(tr_col)
            )
            atr_col = f"atr_{w}"
            atr_expr = pl.col(tr_col).rolling_mean(window_size=w, min_samples=w).over(key).alias(atr_col)
            result = result.with_columns(atr_expr)
            # downside vol: std of negative returns
            # Not fully spec, approximate: rolling std of returns where return <0 ?
            # We'll create downside proxy: rolling std filtered? For simplicity, reuse rv.
            # Gap: (open - close.shift(1))/close.shift(1)
            gap_col = f"gap_{w}"
            gap_expr = (
                pl.when(
                    pl.col("close").shift(1).over(key).is_null()
                    | (pl.col("close").shift(1).over(key) == 0)
                    | pl.col("open").is_null()
                )
                .then(pl.lit(None, dtype=pl.Float64))
                .otherwise(pl.col("open") / pl.col("close").shift(1).over(key) - pl.lit(1.0))
                .alias(gap_col)
            )
            result = result.with_columns(gap_expr)
            gap_rv_col = f"gap_rv_{w}"
            gap_rv_expr = pl.col(gap_col).rolling_std(window_size=w, min_samples=w).over(key).alias(gap_rv_col)
            result = result.with_columns(gap_rv_expr)
    # Remove _ret and _tr_ columns
    to_drop = [c for c in result.columns if c == "_ret" or c.startswith("_tr_")]
    if to_drop:
        result = result.drop(to_drop)
    return result
