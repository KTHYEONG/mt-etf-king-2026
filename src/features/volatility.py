from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from src.features.pit import assert_pit, restrict_to_traded_sessions


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
    # 시장 전체가 무거래인 phantom 세션을 롤링 윈도우에서 제외(add_trend와 동일 방어 패턴).
    calc = restrict_to_traded_sessions(sorted_frame, price_col="close")
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
    calc = calc.with_columns(ret_expr)
    output_cols: list[str] = []
    for w in windows:
        rv_col = f"rv_{w}"
        # rolling std of return
        rv_expr = pl.col("_ret").rolling_std(window_size=w, min_samples=w).over(key).alias(rv_col)
        calc = calc.with_columns(rv_expr)
        output_cols.append(rv_col)
        # ATR-like: (high-low)/close rolling mean
        if "high" in frame.columns and "low" in frame.columns:
            tr_col = f"_tr_{w}"
            # true range approximation: high - low divided by close
            calc = calc.with_columns(
                pl.when(pl.col("close").is_null() | (pl.col("close") == 0))
                .then(pl.lit(None, dtype=pl.Float64))
                .otherwise((pl.col("high") - pl.col("low")) / pl.col("close"))
                .alias(tr_col)
            )
            atr_col = f"atr_{w}"
            atr_expr = pl.col(tr_col).rolling_mean(window_size=w, min_samples=w).over(key).alias(atr_col)
            calc = calc.with_columns(atr_expr)
            output_cols.append(atr_col)
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
            calc = calc.with_columns(gap_expr)
            output_cols.append(gap_col)
            gap_rv_col = f"gap_rv_{w}"
            gap_rv_expr = pl.col(gap_col).rolling_std(window_size=w, min_samples=w).over(key).alias(gap_rv_col)
            calc = calc.with_columns(gap_rv_expr)
            output_cols.append(gap_rv_col)
    if not output_cols:
        return sorted_frame
    # Drop any stale same-named columns from a prior call (e.g. incremental/idempotent
    # re-invocation) so the join overwrites rather than colliding into "<col>_right".
    base = sorted_frame.drop([c for c in output_cols if c in sorted_frame.columns])
    return base.join(calc.select([key, "date", *output_cols]), on=[key, "date"], how="left")
