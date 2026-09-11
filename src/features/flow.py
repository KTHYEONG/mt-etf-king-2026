from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from src.features.pit import assert_pit, restrict_to_traded_sessions


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
    # Ensure decompose columns exist (creation_flow_krw, performance_effect); shift(1)-only,
    # not window-based, so it is unaffected by the phantom-session propagation bug below.
    # Compute creation_flow_krw if not already present
    if "creation_flow_krw" not in result.columns:
        result = decompose_aum_change(result, decision_date, key=key)
    # 시장 전체가 무거래인 phantom 세션을 롤링 윈도우(flow_{w}d, ADV5/ADV20)에서 제외
    # (add_trend/add_volatility와 동일 방어 패턴). price_col="close"는 다른 피처 모듈과
    # 동일 기준으로 phantom 세션을 판별한다(해당 세션은 trading_value도 함께 결측/0이다).
    calc = restrict_to_traded_sessions(result, price_col="close")
    output_cols: list[str] = []
    # flow_ratio = (shares - shares.shift1)/shares.shift1
    if "shares_outstanding" in result.columns:
        sh_prev = pl.col("shares_outstanding").shift(1).over(key)
        flow_ratio_expr = (
            pl.when(sh_prev.is_null() | (sh_prev == 0) | pl.col("shares_outstanding").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise((pl.col("shares_outstanding") - sh_prev) / sh_prev)
            .alias("flow_ratio")
        )
        calc = calc.with_columns(flow_ratio_expr)
        output_cols.append("flow_ratio")
        # cumulative flow over windows
        for w in windows:
            col = f"flow_{w}d"
            # rolling sum of creation_flow_krw
            sum_expr = pl.col("creation_flow_krw").rolling_sum(window_size=w, min_samples=w).over(key).alias(col)
            calc = calc.with_columns(sum_expr)
            output_cols.append(col)
    # turnover = trading_value / net_assets
    if "trading_value" in result.columns and "net_assets" in result.columns:
        turnover_expr = (
            pl.when(pl.col("net_assets").is_null() | (pl.col("net_assets") == 0) | pl.col("trading_value").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("trading_value") / pl.col("net_assets"))
            .alias("turnover")
        )
        calc = calc.with_columns(turnover_expr)
        output_cols.append("turnover")
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
        calc = calc.with_columns(vol_exp_expr)
        output_cols.append("volume_expansion")
    # disparity = (close - nav)/nav
    if "close" in result.columns and "nav" in result.columns:
        disp_expr = (
            pl.when(pl.col("nav").is_null() | (pl.col("nav") == 0) | pl.col("close").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise((pl.col("close") - pl.col("nav")) / pl.col("nav"))
            .alias("disparity")
        )
        calc = calc.with_columns(disp_expr)
        output_cols.append("disparity")
    if not output_cols:
        return result
    # Drop any stale same-named columns from a prior call (e.g. incremental/idempotent
    # re-invocation) so the join overwrites rather than colliding into "<col>_right".
    base = result.drop([c for c in output_cols if c in result.columns])
    return base.join(calc.select([key, "date", *output_cols]), on=[key, "date"], how="left")
