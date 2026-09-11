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


def restrict_to_traded_sessions(frame: pl.DataFrame, price_col: str = "close") -> pl.DataFrame:
    """Drop rows on market-wide phantom sessions (every ticker has a null/non-finite/non-positive
    `price_col` on that date, e.g. an unscheduled exchange closure) before a windowed/rolling
    feature computation.

    `pl.Expr.rolling_*(window_size=w, min_samples=w)` counts *physical rows*, not valid trading
    sessions: a single phantom-session row inside the window makes every window that still
    contains it produce null, silently nulling the feature for the next `w` sessions across the
    entire panel even though `w` real trading sessions are available once the phantom row is
    skipped. Restricting to traded-session rows first (mirroring `add_momentum`'s existing
    phantom-safe pattern) makes the window count real trading sessions only; callers must
    left-join their computed columns back onto the full frame by (key, date) so a phantom
    session's own row still correctly resolves to null (no price => no valid feature that day).
    """
    if frame.height == 0 or price_col not in frame.columns or "date" not in frame.columns:
        return frame
    valid = pl.col(price_col).is_not_null() & pl.col(price_col).is_finite() & (pl.col(price_col) > 0.0)
    traded_dates = frame.filter(valid.any().over("date")).select(pl.col("date").unique()).to_series().to_list()
    return frame.filter(pl.col("date").is_in(traded_dates))
