# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from src.core.paths import DataPaths


def make_backtest_run_id(model: str, start: date, end: date, *, now: datetime | None = None) -> str:
    ts = now if now is not None else datetime.now(UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    # filesystem-safe UTC timestamp without separators that conflict with paths
    # use YYYYMMDDTHHMMSSZ format
    stamp = ts.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    # sanitize model: replace any path separators
    safe_model = str(model).replace("/", "_").replace("\\", "_").strip()
    # build token
    run_id = f"{stamp}_{safe_model}_{start:%Y%m%d}_{end:%Y%m%d}"
    # ensure no path separators remain
    if "/" in run_id or "\\" in run_id:
        run_id = run_id.replace("/", "_").replace("\\", "_")
    return run_id


def write_backtest_result(
    paths: DataPaths,
    *,
    run_id: str,
    meta: Mapping[str, object],
    summary: Mapping[str, object],
    daily: pl.DataFrame | None = None,
    trades: pl.DataFrame | None = None,
    windows: pl.DataFrame | None = None,
) -> Path:
    dest = paths.results(run_id)
    if dest.exists():
        raise FileExistsError(f"results directory already exists: {dest}")
    dest.mkdir(parents=True, exist_ok=False)
    meta_doc = dict(meta)
    summary_doc = dict(summary)
    if daily is not None and trades is not None:
        from src.reporting.timeseries import write_timeseries_parquet, write_window_timeseries

        artifacts = write_timeseries_parquet(dest, daily, trades)
        # wiring: ensure write_window_timeseries imported and invoked when windows provided
        if windows is not None:
            win_path = write_window_timeseries(dest, windows)
            artifacts["windows"] = "windows.parquet"
            _ = win_path
        meta_doc["artifacts"] = artifacts
        summary_doc["artifacts"] = artifacts
        summary_doc["daily_rows"] = int(daily.height)
        summary_doc["trade_rows"] = int(trades.height)
    meta_path = dest / "meta.json"
    summary_path = dest / "summary.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return dest
