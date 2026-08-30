from __future__ import annotations

from pathlib import Path

import polars as pl

_PARQUET_KW = {"compression": "zstd", "statistics": True}


def enrich_daily_timeseries(daily: pl.DataFrame) -> pl.DataFrame:
    """Add cumulative return and drawdown columns for tournament-style daily tracking."""
    if daily.height == 0:
        return daily
    ret_col = "ret" if "ret" in daily.columns else ("return" if "return" in daily.columns else None)
    df = daily.sort("date")
    if "equity" in df.columns:
        eq0 = float(df.select(pl.col("equity").first()).item())  # type: ignore[arg-type]
        if eq0 == 0:
            eq0 = 1.0
        df = df.with_columns(
            (pl.col("equity") / pl.lit(eq0) - 1.0).alias("cum_return"),
            (pl.col("equity") / pl.col("equity").cum_max() - 1.0).alias("drawdown"),
        )
    elif ret_col is not None:
        one_plus = 1.0 + pl.col(ret_col).fill_null(0.0)
        df = df.with_columns(
            one_plus.cum_prod().alias("_eq"),
        ).with_columns(
            (pl.col("_eq") - 1.0).alias("cum_return"),
            (pl.col("_eq") / pl.col("_eq").cum_max() - 1.0).alias("drawdown"),
        ).drop("_eq")
    return df


def normalize_trades_timeseries(trades: pl.DataFrame) -> pl.DataFrame:
    """Ensure trade log has analysis-friendly columns and stable ordering."""
    if trades.height == 0:
        schema = {
            "decision_date": pl.Date,
            "execution_date": pl.Date,
            "ticker": pl.Utf8,
            "side": pl.Utf8,
            "weight_before": pl.Float64,
            "weight_after": pl.Float64,
            "delta_weight": pl.Float64,
            "weight": pl.Float64,
            "price": pl.Float64,
        }
        return pl.DataFrame(schema=schema)
    df = trades
    if "weight_after" not in df.columns and "weight" in df.columns:
        df = df.with_columns(pl.col("weight").alias("weight_after"))
    if "weight_before" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("weight_before"))
    if "delta_weight" not in df.columns:
        df = df.with_columns((pl.col("weight_after") - pl.col("weight_before")).alias("delta_weight"))
    if "side" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("delta_weight") > 0)
            .then(pl.lit("BUY"))
            .when(pl.col("delta_weight") < 0)
            .then(pl.lit("SELL"))
            .otherwise(pl.lit("HOLD"))
            .alias("side")
        )
    sort_cols = [c for c in ("execution_date", "decision_date", "ticker") if c in df.columns]
    if sort_cols:
        df = df.sort(sort_cols)
    keep = [
        c
        for c in (
            "decision_date",
            "execution_date",
            "ticker",
            "side",
            "weight_before",
            "weight_after",
            "delta_weight",
            "weight",
            "price",
        )
        if c in df.columns
    ]
    return df.select(keep)


def write_timeseries_parquet(dest: Path, daily: pl.DataFrame, trades: pl.DataFrame) -> dict[str, str]:
    """Write compact zstd Parquet artifacts under a backtest result directory."""
    dest.mkdir(parents=True, exist_ok=True)
    daily_out = enrich_daily_timeseries(daily)
    trades_out = normalize_trades_timeseries(trades)
    daily_path = dest / "daily.parquet"
    trades_path = dest / "trades.parquet"
    daily_out.write_parquet(daily_path, **_PARQUET_KW)
    trades_out.write_parquet(trades_path, **_PARQUET_KW)
    return {"daily": daily_path.name, "trades": trades_path.name}
