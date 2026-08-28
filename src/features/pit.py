from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl


class PitViolationError(ValueError):
    pass


def assert_pit(frame: pl.DataFrame, decision_date: date, date_column: str = "date") -> pl.DataFrame:
    if frame.height == 0:
        return frame
    if date_column not in frame.columns:
        return frame
    # Filter rows where date > decision_date
    # Handle Date type comparison
    violating = frame.filter(pl.col(date_column) > decision_date)
    if violating.height > 0:
        # collect up to 5 offending dates
        dates = violating.select(pl.col(date_column)).to_series().to_list()
        # unique sorted
        uniq = sorted({d for d in dates if d is not None})
        sample = uniq[:5]
        # format as strings
        sample_str = ", ".join(str(d) for d in sample)
        raise PitViolationError(f"PIT violation: dates {sample_str} exceed decision_date {decision_date}")
    return frame


def align_session_grid(frame: pl.DataFrame, sessions: Sequence[date], key: str = "ticker") -> pl.DataFrame:
    if frame.height == 0:
        return frame
    if key not in frame.columns or "date" not in frame.columns:
        return frame
    # Ensure sessions sorted
    sorted_sessions = sorted(sessions)
    # Determine tickers
    tickers = frame.select(pl.col(key).unique()).to_series().to_list()
    # Build grid rows
    grid_rows: list[dict[str, object]] = []
    # Precompute frame min/max per ticker for quick lookup
    # Use group_by to get min/max
    grouped = frame.group_by(key).agg([
        pl.col("date").min().alias("first_seen"),
        pl.col("date").max().alias("last_seen"),
    ])
    first_map: dict[str, date] = {}
    last_map: dict[str, date] = {}
    for row in grouped.iter_rows(named=True):
        t = str(row[key])
        first_map[t] = row["first_seen"]
        last_map[t] = row["last_seen"]
    for ticker in tickers:
        tstr = str(ticker)
        f = first_map.get(tstr)
        last = last_map.get(tstr)
        if f is None or last is None:
            continue
        # sessions in [f, last]
        sub = [s for s in sorted_sessions if f <= s <= last]
        grid_rows.extend({key: tstr, "date": s} for s in sub)
    if not grid_rows:
        return frame
    grid = pl.DataFrame(grid_rows)
    # Ensure date type is Date
    grid = grid.with_columns(pl.col("date").cast(pl.Date))
    # Original frame may have date not as Date? Ensure cast
    # Left join grid with frame on key+date
    # Need to keep all grid rows, attach original columns
    result = grid.join(frame, on=[key, "date"], how="left")
    # Sort
    result = result.sort([key, "date"])
    return result
