from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from src.core.paths import DataPaths

# Minimal backtest panel columns: subset of gold/silver, not the full gold schema.
# Loader intersects with available columns; never permanently drops gold columns on disk.
BACKTEST_PANEL_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "close",
    "open",
    "trading_value",
    "is_tradable",
    "name",
    "underlying_index_name",
    "mom_3",
    "mom_5",
    "mom_10",
    "mom_20",
    "mom_40",
    "mom_60",
    "ma_20",
    "ma_ratio_20",
    "drawdown_20",
    "volume_expansion",
)


def align_feature_rows(silver: pl.DataFrame, gold: pl.DataFrame) -> pl.DataFrame:
    keys = ["date", "ticker"]
    for frame in (silver, gold):
        counts = frame.group_by(keys).len()
        if (counts["len"] > 1).any():
            raise ValueError("duplicate (date,ticker) keys")
    silver_keys = silver.select(keys).unique()
    gold_keys = gold.select(keys).unique()
    extra = gold_keys.join(silver_keys, on=keys, how="anti")
    if extra.height > 0:
        placeholders = gold.join(extra, on=keys, how="semi")
        ohlc = [c for c in ("open", "high", "low", "close") if c in gold.columns]
        ok = True
        for row in placeholders.iter_rows(named=True):
            ohlc_null = all(row.get(c) is None for c in ohlc)
            tradable = row.get("is_tradable")
            tv = row.get("trading_value")
            tv_zero = tv is None or (isinstance(tv, float) and tv != tv) or tv == 0
            if not (ohlc_null and tradable in (False, None) and tv_zero):
                ok = False
                break
        if not ok:
            raise ValueError("gold-only quote rows")
    missing = silver_keys.join(gold_keys, on=keys, how="anti")
    if missing.height > 0:
        raise ValueError("missing gold rows")
    joined = silver.select([*keys, "open", "close", "is_tradable", "trading_value"]).join(gold.select([*keys, "open", "close", "is_tradable", "trading_value"]), on=keys, how="inner", suffix="_gold")
    for col in ("open", "close", "is_tradable", "trading_value"):
        left, right = col, f"{col}_gold"
        mismatch = joined.filter(~((pl.col(left) == pl.col(right)) | (pl.col(left).is_null() & pl.col(right).is_null())))
        if mismatch.height > 0:
            raise ValueError(f"quote mismatch in {col}")
    return gold.join(silver_keys, on=keys, how="semi")


def load_backtest_panel(
    paths: DataPaths,
    *,
    columns: Sequence[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame | None:
    requested = list(columns) if columns is not None else list(BACKTEST_PANEL_COLUMNS)
    silver_path = paths.silver("etf_daily")
    gold_path = paths.gold("etf_features")
    if silver_path.exists() and gold_path.exists():
        silver_full = pl.read_parquet(str(silver_path))
        gold_full = pl.read_parquet(str(gold_path))
        aligned = align_feature_rows(silver_full, gold_full)
        projection = [c for c in requested if c in aligned.columns]
        if start is not None:
            aligned = aligned.filter(pl.col("date") >= start)
        if end is not None:
            aligned = aligned.filter(pl.col("date") <= end)
        return aligned.select(projection)
    # Prefer gold then silver
    candidates = [
        paths.gold("etf_features"),
        paths.silver("etf_daily"),
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            schema = pl.scan_parquet(str(p)).collect_schema()
            file_columns = list(schema.names())
        except Exception:  # noqa: S112
            try:
                # fallback: read header quickly
                df_tmp = pl.read_parquet(str(p), n_rows=0)  # noqa: S112
                file_columns = list(df_tmp.columns)  # noqa: S112
            except Exception:  # noqa: S112
                continue  # noqa: S112
        projection = [c for c in requested if c in file_columns]
        # If nothing intersects, this file not usable; try next
        if not projection:
            continue
        # need date for filtering even if not requested
        need_date = (start is not None or end is not None) and "date" in file_columns and "date" not in projection
        cols_to_read = [*projection, "date"] if need_date else projection  # noqa: RUF005
        try:
            df = pl.read_parquet(str(p), columns=cols_to_read)  # noqa: S112
        except Exception:  # noqa: S112
            continue  # noqa: S112
        # inclusive date filter
        if (start is not None or end is not None) and "date" in df.columns:
            if start is not None:
                df = df.filter(pl.col("date") >= start)
            if end is not None:
                df = df.filter(pl.col("date") <= end)
            if need_date and "date" not in requested:
                # drop the temporarily added date to keep output subset of requested
                import contextlib

                with contextlib.suppress(Exception):  # noqa: SIM105
                    df = df.drop("date")  # noqa: SIM105
        # Never mutate on-disk; projection already enforced
        return df
    return None
