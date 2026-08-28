from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from src.core.paths import DataPaths
from src.data.bronze import BronzeStore
from src.data.providers.krx import resolve_endpoint
from src.data.schema import ETF_DAILY_SCHEMA, INDEX_DAILY_SCHEMA, STOCK_DAILY_SCHEMA
from src.data.validation import PanelValidator, ValidationReport, classify_session, mark_tradability

logger = logging.getLogger(__name__)

SCHEMAS = {
    ETF_DAILY_SCHEMA.name: ETF_DAILY_SCHEMA,
    INDEX_DAILY_SCHEMA.name: INDEX_DAILY_SCHEMA,
    STOCK_DAILY_SCHEMA.name: STOCK_DAILY_SCHEMA,
}


@dataclass(frozen=True)
class BuildResult:
    dataset: str
    path: Path
    rows: int
    sessions: int
    report: ValidationReport


class SilverBuilder:
    def __init__(self, store: BronzeStore, paths: DataPaths, validator: PanelValidator) -> None:
        self._store = store
        self._paths = paths
        self._validator = validator

    def load(self, dataset: str) -> pl.DataFrame:
        path = self._paths.silver(dataset)
        return pl.read_parquet(path)

    def build(self, dataset: str, mode: Literal["full", "incremental"] = "incremental") -> BuildResult:
        if dataset not in SCHEMAS:
            raise KeyError(f"unknown dataset {dataset}")
        schema = SCHEMAS[dataset]
        endpoint = schema.endpoint
        # Also resolve via provider mapping to ensure wiring reference (if dataset is etf_daily, endpoint is same)
        import contextlib

        with contextlib.suppress(KeyError):
            _ = resolve_endpoint(dataset)

        silver_path = self._paths.silver(dataset)

        # Determine base sessions
        all_sessions = self._store.available_sessions(endpoint)
        all_sessions.sort()

        # Determine incremental filtering
        existing_frame: pl.DataFrame | None = None
        max_existing_date = None
        if mode == "incremental" and silver_path.exists():
            try:
                existing_frame = pl.read_parquet(silver_path)
                if existing_frame.height > 0 and "date" in existing_frame.columns:
                    max_existing_date = existing_frame.select(pl.col("date").max()).item()
            except Exception:
                existing_frame = None
                max_existing_date = None

        # Filter sessions to process
        if mode == "incremental" and max_existing_date is not None:
            sessions_to_process = [d for d in all_sessions if d > max_existing_date]
        else:
            sessions_to_process = all_sessions

        # Price field for classify_session: source_field of close column
        price_field = None
        for col in schema.columns:
            if col.column == "close":
                price_field = col.source_field
                break
        if price_field is None:
            price_field = "TDD_CLSPRC"

        decoded_rows: list[dict[str, object]] = []
        processed_sessions: list[object] = []
        skipped_sessions: list[object] = []

        for bas_dd in sessions_to_process:
            try:
                record = self._store.read(endpoint, bas_dd)
            except Exception:
                skipped_sessions.append(bas_dd)
                continue
            rec_rows = record.rows
            is_trading = classify_session(rec_rows, price_field, min_valid_ratio=0.5)
            if not is_trading:
                skipped_sessions.append(bas_dd)
                continue
            # Decode
            try:
                decoded = schema.decode_rows(rec_rows)
            except Exception as exc:
                logger.error(f"[DATA] decode failed dataset={dataset} date={bas_dd} error={exc!r}")
                raise
            decoded_rows.extend(decoded)
            processed_sessions.append(bas_dd)

        # Build new frame from decoded rows
        if decoded_rows:
            new_frame = pl.DataFrame(decoded_rows)
            # Ensure date column is Date type
            if "date" in new_frame.columns:
                new_frame = new_frame.with_columns(pl.col("date").cast(pl.Date))
        else:
            # Empty frame with schema columns + is_tradable will be added later
            # Create empty frame with schema columns
            new_frame = pl.DataFrame([])

        # If incremental and existing_frame exists, concatenate
        if existing_frame is not None and mode == "incremental":
            if new_frame.height > 0:
                # Ensure columns align: existing_frame already has is_tradable
                # new_frame not yet has is_tradable; will be added after concat? Better add after concat via mark_tradability on combined?
                # To keep logic simple, concat raw then apply mark_tradability on combined
                combined = pl.concat([existing_frame, new_frame], how="diagonal")
            else:
                combined = existing_frame
        else:
            combined = new_frame

        # If combined empty, create empty with correct dtypes?
        if combined.height == 0:
            # Still need to produce empty frame with schema columns plus is_tradable?
            # Create empty frame with schema
            # For now, just ensure is_tradable column exists via mark_tradability handling empty case
            frame = mark_tradability(combined, max_abs_disparity=0.20)
        else:
            # Need to ensure is_tradable not already present for new_frame part; if combined from incremental, existing already had is_tradable, new doesn't
            # Remove is_tradable if present to recompute consistently for whole panel (robust outlier needs full distribution)
            if "is_tradable" in combined.columns:
                combined = combined.drop("is_tradable")
            # Apply mark_tradability (uses Polars expressions, median/MAD)
            frame = mark_tradability(combined, max_abs_disparity=0.20)
            # Sort by date and ticker for idempotence
            sort_cols = []
            if "date" in frame.columns:
                sort_cols.append("date")
            if "ticker" in frame.columns:
                sort_cols.append("ticker")
            elif "index_name" in frame.columns:
                sort_cols.append("index_name")
            if sort_cols:
                frame = frame.sort(sort_cols)

        # Validate
        report = self._validator.validate(dataset, frame)

        # Handle fatal: MUST NOT write, MUST raise
        if report.is_fatal():
            # Log with [DATA] tag
            logger.error(f"[DATA] build status=fail dataset={dataset} reason=fatal_validation rows={frame.height}")
            # Ensure file does not exist afterwards as per requirement if it was not existing before?
            # If mode is full and report fatal, remove file if it was newly created? But we haven't written yet, so just ensure not created.
            # If incremental and existing file existed, should we keep it? The test expects after fatal build, file does NOT exist (implies no prior file)
            # For safety, if silver_path exists and we are in full mode and fatal, we could remove? But requirement says MUST NOT write when fatal, so we should not overwrite.
            # If incremental fatal, keep existing? But test for 03-08 says building full then adding one later bronze session and building incremental... When validator returns fatal report, build raises and DataPaths.silver does not exist afterwards.
            # That test case is isolated: start with no file, validator fatal -> build raises, file not exist.
            # So we just raise without writing.
            raise RuntimeError(f"validation fatal for {dataset}: {[i.gate for i in report.issues if i.severity.value == 'CRITICAL']}")

        # Write Parquet with zstd via pyarrow
        # Must be written to DataPaths.silver(dataset)
        silver_path.parent.mkdir(parents=True, exist_ok=True)
        # Use Polars write_parquet with compression zstd and pyarrow
        # Polars uses compression param and use_pyarrow flag
        try:
            frame.write_parquet(str(silver_path), compression="zstd", use_pyarrow=True)
        except TypeError:
            # Fallback without use_pyarrow if version differs
            frame.write_parquet(str(silver_path), compression="zstd")

        n_rows = frame.height
        n_sessions = 0
        if "date" in frame.columns and n_rows > 0:
            try:
                n_sessions = int(frame.select(pl.col("date").n_unique()).item())
            except Exception:
                n_sessions = len(frame.select(pl.col("date").unique()).to_series().to_list())

        # Log summary with [DATA] tag and truncated instrument list handling already in validator
        logger.info(f"[DATA] build dataset={dataset} mode={mode} rows={n_rows} sessions={n_sessions} path={silver_path}")

        return BuildResult(dataset=dataset, path=silver_path, rows=n_rows, sessions=n_sessions, report=report)
