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

    _append_to_registries(paths, run_id=run_id, meta=meta_doc, summary=summary_doc)
    return dest


def _extract_summary_record(run_id: str, meta: Mapping[str, object], summary: Mapping[str, object]) -> dict[str, object]:
    quantiles = summary.get("quantiles") or {}
    exceedance = summary.get("exceedance") or {}
    realised = summary.get("realised_exposure") or {}
    field_rel = summary.get("field_relative") or {}
    capture = summary.get("capture") or {}
    attainability = summary.get("attainability") or {}
    n_attainable = summary.get("n_attainable") or {}

    def _f(val: object) -> float | None:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _i(val: object) -> int | None:
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _c(val: object) -> float | None:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    return {
        "run_id": str(run_id),
        "model": str(meta.get("model") or meta.get("strategy_id") or ""),
        "strategy_id": str(meta.get("strategy_id") or ""),
        "legacy_model_id": str(meta.get("legacy_model_id") or ""),
        "start": str(meta.get("start") or ""),
        "end": str(meta.get("end") or ""),
        "horizon": _i(meta.get("horizon")),
        "n_windows": _i(summary.get("n_windows")),
        "n_effective": _i(summary.get("n_effective")),
        "p_gt_30": _f(exceedance.get("0.3")),
        "p_gt_40": _f(exceedance.get("0.4")),
        "p_gt_50": _f(exceedance.get("0.5")),
        "q50": _f(quantiles.get("0.5")),
        "q90": _f(quantiles.get("0.9")),
        "q95": _f(quantiles.get("0.95")),
        "q99": _f(quantiles.get("0.99")),
        "cvar_05": _f(summary.get("cvar_05")),
        "giveback_median": _f(summary.get("giveback_median")),
        "giveback_q90": _f(summary.get("giveback_q90")),
        "right_tail_score": _f(summary.get("right_tail_score")),
        "win_rate": _f(field_rel.get("win_rate")),
        "top2_rate": _f(field_rel.get("top2_rate")),
        "effective_gross_mean": _f(realised.get("effective_gross_mean")),
        "effective_gross_max": _f(summary.get("effective_gross_max") or realised.get("effective_gross_max")),
        "gross_violation_count": _i(summary.get("gross_violation_count") or realised.get("gross_violation_count") or 0),
        "turnover": _f(realised.get("turnover")),
        "championship_gate_status": str(summary.get("championship_gate_status") or ""),
        "adoption_gate_status": str(summary.get("adoption_gate_status") or ""),
        "objective_gate_status": str(summary.get("objective_gate_status") or ""),
        "capture_30": _c(capture.get("0.3")),
        "capture_40": _c(capture.get("0.4")),
        "capture_50": _c(capture.get("0.5")),
        "capture_60": _c(capture.get("0.6")),
        "attainable_30": _i(n_attainable.get("0.3")),
        "attainable_40": _i(n_attainable.get("0.4")),
        "attainable_50": _i(n_attainable.get("0.5")),
        "attainable_60": _i(n_attainable.get("0.6")),
        "attainability_30": _c(attainability.get("0.3")),
        "attainability_40": _c(attainability.get("0.4")),
        "attainability_50": _c(attainability.get("0.5")),
        "attainability_60": _c(attainability.get("0.6")),
        "breadth_mean": _c(summary.get("breadth_mean")),
        "created_at": str(meta.get("created_at") or datetime.now(UTC).isoformat()),
    }


def _append_to_registries(paths: DataPaths, *, run_id: str, meta: Mapping[str, object], summary: Mapping[str, object]) -> None:
    record = _extract_summary_record(run_id, meta, summary)
    anchor = paths._anchor_root()

    # 1. Append to docs/results/runs_registry.jsonl (Git tracked)
    jsonl_dir = anchor / "docs/results"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / "runs_registry.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2. Append/Upsert into data/results/runs.parquet (Local analytics table)
    data_results_dir = paths.root / "results"
    data_results_dir.mkdir(parents=True, exist_ok=True)
    runs_pq_path = data_results_dir / "runs.parquet"

    new_row = pl.DataFrame([record])
    if runs_pq_path.exists():
        try:
            existing = pl.read_parquet(runs_pq_path)
            # filter out existing run_id to support idempotent upsert
            existing = existing.filter(pl.col("run_id") != run_id)
            combined = pl.concat([existing, new_row], how="diagonal")
        except Exception:
            combined = new_row
    else:
        combined = new_row
    combined.write_parquet(runs_pq_path, compression="zstd")


def rebuild_runs_registry(paths: DataPaths) -> int:
    """Scan docs/results directories and rebuild runs_registry.jsonl and runs.parquet."""
    anchor = paths._anchor_root()
    docs_results = anchor / "docs/results"
    if not docs_results.exists():
        return 0

    candidate_dirs = []
    data_results_dir = paths.root / "results"
    if data_results_dir.exists():
        candidate_dirs.extend([d for d in data_results_dir.iterdir() if d.is_dir()])
    if docs_results.exists():
        candidate_dirs.extend([d for d in docs_results.iterdir() if d.is_dir()])

    seen_run_ids = set()
    records: list[dict[str, object]] = []
    # Find all directories containing summary.json and meta.json
    for child in sorted(candidate_dirs, key=lambda p: p.name):
        if child.name in seen_run_ids:
            continue
        meta_file = child / "meta.json"
        summ_file = child / "summary.json"
        if not meta_file.exists() or not summ_file.exists():
            continue
        try:
            with meta_file.open(encoding="utf-8") as f:
                meta = json.load(f)
            with summ_file.open(encoding="utf-8") as f:
                summ = json.load(f)
            record = _extract_summary_record(child.name, meta, summ)
            records.append(record)
            seen_run_ids.add(child.name)
        except Exception:
            continue

    if not records:
        return 0

    # Write runs_registry.jsonl
    jsonl_path = docs_results / "runs_registry.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write runs.parquet
    data_results_dir = paths.root / "results"
    data_results_dir.mkdir(parents=True, exist_ok=True)
    runs_pq_path = data_results_dir / "runs.parquet"
    df = pl.DataFrame(records)
    df.write_parquet(runs_pq_path, compression="zstd")
    return len(records)

