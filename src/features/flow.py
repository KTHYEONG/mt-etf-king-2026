from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from src.features.pit import assert_pit


def decompose_aum_change(
    frame: pl.DataFrame,
    decision_date: date,
    key: str = "ticker",
) -> pl.DataFrame:
    assert_pit(frame, decision_date)
    if frame.height == 0:
        return frame
    if "shares_outstanding" not in frame.columns or "nav" not in frame.columns:
        return frame
    sorted_frame = frame.sort([key, "date"])
    result = sorted_frame
    # shift values per ticker
    sh_shares = pl.col("shares_outstanding").shift(1).over(key)
    sh_nav = pl.col("nav").shift(1).over(key)
    creation = (
        pl.when(pl.col("shares_outstanding").is_null() | pl.col("nav").is_null() | sh_shares.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise((pl.col("shares_outstanding") - sh_shares) * pl.col("nav"))
        .alias("creation_flow_krw")
    )
    perf = (
        pl.when(sh_shares.is_null() | pl.col("nav").is_null() | sh_nav.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(sh_shares * (pl.col("nav") - sh_nav))
        .alias("performance_effect")
    )
    result = result.with_columns([creation, perf])
    return result


def add_flow(
    frame: pl.DataFrame,
    windows: Sequence[int],
    decision_date: date,
    key: str = "ticker",
) -> pl.DataFrame:
    assert_pit(frame, decision_date)
    if frame.height == 0:
        return frame
    sorted_frame = frame.sort([key, "date"])
    result = sorted_frame
    # Ensure decompose columns exist (creation_flow_krw, performance_effect)
    # Compute creation_flow_krw if not already present
    if "creation_flow_krw" not in result.columns:
        result = decompose_aum_change(result, decision_date, key=key)
    # flow_ratio = (shares - shares.shift1)/shares.shift1
    if "shares_outstanding" in result.columns:
        sh_prev = pl.col("shares_outstanding").shift(1).over(key)
        flow_ratio_expr = (
            pl.when(sh_prev.is_null() | (sh_prev == 0) | pl.col("shares_outstanding").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise((pl.col("shares_outstanding") - sh_prev) / sh_prev)
            .alias("flow_ratio")
        )
        result = result.with_columns(flow_ratio_expr)
        # cumulative flow over windows
        for w in windows:
            col = f"flow_{w}d"
            # rolling sum of creation_flow_krw
            sum_expr = pl.col("creation_flow_krw").rolling_sum(window_size=w, min_samples=w).over(key).alias(col)
            result = result.with_columns(sum_expr)
    # turnover = trading_value / net_assets
    if "trading_value" in result.columns and "net_assets" in result.columns:
        turnover_expr = (
            pl.when(pl.col("net_assets").is_null() | (pl.col("net_assets") == 0) | pl.col("trading_value").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("trading_value") / pl.col("net_assets"))
            .alias("turnover")
        )
        result = result.with_columns(turnover_expr)
        # volume_expansion = ADV5 / ADV20 ; need ADV windows fixed 5 and 20 as per spec
        # Use trading_value rolling mean
        # Ensure windows contains 5 and 20 for expansion; if not, still compute using 5 and 20
        adv_short = min(windows)
        adv_long = max(windows)
        adv5 = pl.col("trading_value").cast(pl.Float64).rolling_mean(window_size=adv_short, min_samples=adv_short).over(key)
        adv20 = pl.col("trading_value").cast(pl.Float64).rolling_mean(window_size=adv_long, min_samples=adv_long).over(key)
        vol_exp_expr = (
            pl.when(adv20.is_null() | (adv20 == 0))
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(adv5 / adv20)
            .alias("volume_expansion")
        )
        result = result.with_columns(vol_exp_expr)
    # disparity = (close - nav)/nav
    if "close" in result.columns and "nav" in result.columns:
        disp_expr = (
            pl.when(pl.col("nav").is_null() | (pl.col("nav") == 0) | pl.col("close").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise((pl.col("close") - pl.col("nav")) / pl.col("nav"))
            .alias("disparity")
        )
        result = result.with_columns(disp_expr)
    return result
