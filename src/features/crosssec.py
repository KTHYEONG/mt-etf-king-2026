from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from src.features.pit import assert_pit


def percentile_rank(
    frame: pl.DataFrame,
    columns: Sequence[str],
    decision_date: date,
    by: str = "date",
    suffix: str = "_rs",
) -> pl.DataFrame:
    assert_pit(frame, decision_date)
    if frame.height == 0:
        return frame
    result = frame
    for col in columns:
        if col not in result.columns:
            continue
        out_col = f"{col}{suffix}"
        # count per group: number of non-null values
        # Use window functions
        # n = count(col) over by
        # rank_min = rank(method='min') over by sorted by col ascending
        # percentile = (rank_min -1)/(n-1) when n>1 else null
        n_expr = pl.col(col).count().over(by)
        rank_expr = pl.col(col).rank(method="min").over(by)
        pr_expr = (
            pl.when(pl.col(col).is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .when(n_expr <= 1)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise((rank_expr - pl.lit(1.0)) / (n_expr - pl.lit(1.0)))
            .alias(out_col)
        )
        result = result.with_columns(pr_expr)
    return result


def cross_sectional_zscore(
    frame: pl.DataFrame,
    columns: Sequence[str],
    decision_date: date,
    by: str = "date",
    suffix: str = "_z",
) -> pl.DataFrame:
    assert_pit(frame, decision_date)
    if frame.height == 0:
        return frame
    result = frame
    for col in columns:
        if col not in result.columns:
            continue
        out_col = f"{col}{suffix}"
        mean_expr = pl.col(col).mean().over(by)
        std_expr = pl.col(col).std(ddof=0).over(by)
        z_expr = (
            pl.when(pl.col(col).is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .when(std_expr.is_null() | (std_expr == 0))
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise((pl.col(col) - mean_expr) / std_expr)
            .alias(out_col)
        )
        result = result.with_columns(z_expr)
    return result


def add_momentum_acceleration(
    frame: pl.DataFrame,
    decision_date: date,
    fast: int = 5,
    slow: int = 20,
) -> pl.DataFrame:
    assert_pit(frame, decision_date)
    if frame.height == 0:
        return frame
    # expects mom_fast_rs and mom_slow_rs exist, or compute them via percentile_rank
    fast_col = f"mom_{fast}"
    slow_col = f"mom_{slow}"
    fast_rs = f"{fast_col}_rs"
    slow_rs = f"{slow_col}_rs"
    result = frame
    # If rs columns not present but mom columns present, compute percentile ranks
    need_pr = []
    if fast_col in result.columns and fast_rs not in result.columns:
        need_pr.append(fast_col)
    if slow_col in result.columns and slow_rs not in result.columns:
        need_pr.append(slow_col)
    if need_pr:
        result = percentile_rank(result, need_pr, decision_date, by="date", suffix="_rs")
    if fast_rs in result.columns and slow_rs in result.columns:
        acc_expr = (
            pl.when(pl.col(fast_rs).is_null() | pl.col(slow_rs).is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col(fast_rs) - pl.col(slow_rs))
            .alias("mom_accel")
        )
        result = result.with_columns(acc_expr)
    else:
        # fallback: if mom columns only, compute diff of mom directly? But spec requires rs diff.
        # Emit null column
        result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias("mom_accel"))
    return result
