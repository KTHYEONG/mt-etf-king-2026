from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from src.features.pit import assert_pit, restrict_to_traded_sessions


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
    # 시장 전체가 무거래인 phantom 세션(임시공휴일 등)을 롤링 윈도우에서 제외한다.
    # min_samples=w는 물리적 행 수를 세므로, phantom 세션 1건이 이후 w 세션 동안
    # 전 종목의 지표를 결측 전파시킨다(add_momentum과 동일한 방어 패턴 재사용).
    calc = restrict_to_traded_sessions(sorted_frame, price_col="close")
    output_cols: list[str] = []
    # MA and ratio / slope
    for w in ma_windows:
        ma_col = f"ma_{w}"
        ratio_col = f"ma_ratio_{w}"
        slope_col = f"ma_slope_{w}"
        # rolling mean on traded-session-only grid
        ma_expr = pl.col("close").rolling_mean(window_size=w, min_samples=w).over(key).alias(ma_col)
        calc = calc.with_columns(ma_expr)
        # ratio close / ma
        calc = calc.with_columns(
            pl.when(pl.col(ma_col).is_null() | (pl.col(ma_col) == 0) | pl.col("close").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("close") / pl.col(ma_col))
            .alias(ratio_col)
        )
        # slope: ma / ma.shift(1) -1
        ma_lag = pl.col(ma_col).shift(1).over(key)
        calc = calc.with_columns(
            pl.when(ma_lag.is_null() | (ma_lag == 0) | pl.col(ma_col).is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col(ma_col) / ma_lag - pl.lit(1.0))
            .alias(slope_col)
        )
        output_cols.extend([ma_col, ratio_col, slope_col])
    # Breakout: close vs rolling max, and drawdown
    for w in breakout_windows:
        high_col = f"roll_max_{w}"
        breakout_col = f"breakout_{w}"
        dd_col = f"drawdown_{w}"
        max_expr = pl.col("close").rolling_max(window_size=w, min_samples=w).over(key).alias(high_col)
        calc = calc.with_columns(max_expr)
        # breakout = close / roll_max -1 ? or binary? Use ratio -1
        calc = calc.with_columns(
            pl.when(pl.col(high_col).is_null() | (pl.col(high_col) == 0) | pl.col("close").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("close") / pl.col(high_col) - pl.lit(1.0))
            .alias(breakout_col)
        )
        calc = calc.with_columns(
            pl.when(pl.col(high_col).is_null() | (pl.col(high_col) == 0) | pl.col("close").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise((pl.col("close") - pl.col(high_col)) / pl.col(high_col))
            .alias(dd_col)
        )
        output_cols.extend([high_col, breakout_col, dd_col])
    if not output_cols:
        return sorted_frame
    # Drop any stale same-named columns from a prior call (e.g. incremental/idempotent
    # re-invocation) so the join overwrites rather than colliding into "<col>_right".
    base = sorted_frame.drop([c for c in output_cols if c in sorted_frame.columns])
    return base.join(calc.select([key, "date", *output_cols]), on=[key, "date"], how="left")
