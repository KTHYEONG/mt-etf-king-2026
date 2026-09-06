# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class WindowAttribution:
    window_start: date
    window_end: date
    realized_return: float
    giveback: float
    best_family: str | None
    best_family_return: float
    actual_family: str | None
    actual_family_bh_return: float
    selection_loss: float
    entry_timing_loss: float
    exit_timing_loss: float
    giveback_loss: float
    dominant_bucket: str
    r_actual: float = 0.0
    r_b: float = 0.0
    r_c: float = 0.0
    r_d: float = 0.0
    oracle_gap: float = 0.0


@dataclass(frozen=True)
class TailAttributionSummary:
    n_windows_total: int
    n_analyzed: int
    mean_selection_loss: float
    mean_entry_timing_loss: float
    mean_exit_timing_loss: float
    mean_giveback_loss: float
    selection_dominates_timing: bool
    primary_gap: str
    bucket_counts: dict[str, int]
    windows: tuple[WindowAttribution, ...]
    median_selection_loss: float = 0.0
    median_entry_timing_loss: float = 0.0
    median_exit_timing_loss: float = 0.0
    trimmed_mean_selection_loss: float = 0.0
    trimmed_mean_entry_timing_loss: float = 0.0
    trimmed_mean_exit_timing_loss: float = 0.0
    q75_selection_loss: float = 0.0
    q75_entry_timing_loss: float = 0.0
    q75_exit_timing_loss: float = 0.0
    q90_selection_loss: float = 0.0
    q90_entry_timing_loss: float = 0.0
    q90_exit_timing_loss: float = 0.0
    share_selection: float = 0.0
    share_entry_timing: float = 0.0
    share_exit_timing: float = 0.0
    era_means: dict[str, dict[str, float]] = field(default_factory=dict)


def _close_on(panel: pl.DataFrame, ticker: str, day: date) -> float | None:
    try:
        if panel is None or panel.height == 0:
            return None
        if "date" not in panel.columns or "ticker" not in panel.columns or "close" not in panel.columns:
            return None
        filt = panel.filter((pl.col("ticker") == ticker) & (pl.col("date") == day))
        if filt.height == 0:
            return None
        val = filt["close"][0]
        if val is None:
            return None
        try:
            fval = float(val)
        except Exception:
            return None
        import math

        if not math.isfinite(fval):
            return None
        return fval
    except Exception:
        return None


def compound_close_return(panel: pl.DataFrame, ticker: str, start: date, end: date) -> float | None:
    try:
        c0 = _close_on(panel, ticker, start)
        c1 = _close_on(panel, ticker, end)
        if c0 is None or c1 is None:
            return None
        if c0 <= 0:
            return None
        return float(c1) / float(c0) - 1.0
    except Exception:
        return None


def _open_on(panel: pl.DataFrame, ticker: str, day: date) -> float | None:
    try:
        if panel is None or panel.height == 0:
            return None
        if "date" not in panel.columns or "ticker" not in panel.columns or "open" not in panel.columns:
            return None
        filt = panel.filter((pl.col("ticker") == ticker) & (pl.col("date") == day))
        if filt.height == 0:
            return None
        val = filt["open"][0]
        if val is None:
            return None
        try:
            fval = float(val)
        except Exception:
            return None
        import math

        if not math.isfinite(fval) or fval <= 0:
            return None
        return fval
    except Exception:
        return None


def _next_session_on_or_after(sessions: Sequence[date], decision: date) -> date | None:
    try:
        ordered = sorted(sessions)
    except Exception:
        return None
    for s in ordered:
        try:
            if s > decision:
                return s
        except Exception:
            continue
    return None


def _next_session_after(sessions: Sequence[date], decision: date) -> date | None:
    return _next_session_on_or_after(sessions, decision)


def next_open_path_return(
    panel: pl.DataFrame,
    ticker: str,
    decision_entry: date,
    mark_end: date,
    sessions: Sequence[date],
) -> float | None:
    try:
        fill = _next_session_on_or_after(sessions, decision_entry)
        if fill is None:
            return None
        o = _open_on(panel, ticker, fill)
        c = _close_on(panel, ticker, mark_end)
        if o is None or c is None:
            return None
        return float(c) / float(o) - 1.0
    except Exception:
        return None


def oracle_peak_path_return(
    panel: pl.DataFrame,
    ticker: str,
    decision_entry: date,
    window_end: date,
    sessions: Sequence[date],
) -> float | None:
    try:
        fill = _next_session_on_or_after(sessions, decision_entry)
        if fill is None:
            return None
        o = _open_on(panel, ticker, fill)
        if o is None:
            return None
        try:
            exits = [s for s in list(sessions) if fill <= s <= window_end]
        except Exception:
            return None
        peak: float | None = None
        for e in exits:
            c = _close_on(panel, ticker, e)
            if c is None:
                continue
            r = float(c) / float(o) - 1.0
            if peak is None or r > peak:
                peak = r
        return peak
    except Exception:
        return None


def pit_plus2_tickers(
    master: object,
    *,
    window_start: date,
    universe: object | None = None,
    filters: object | None = None,
) -> list[str]:
    try:
        attrs = getattr(master, "attributes", None)
        if attrs is None:
            return []
        items = attrs.items() if hasattr(attrs, "items") else []
        plus2: list[str] = []
        for k, v in items:
            try:
                if int(getattr(v, "leverage_multiple")) == 2:
                    plus2.append(str(k))
            except Exception:
                continue
        if universe is None:
            return sorted(plus2)
        try:
            snap = universe.get(window_start, filters)  # type: ignore[union-attr]
            allowed = set(snap.tickers)
        except Exception:
            return []
        return sorted(t for t in plus2 if t in allowed)
    except Exception:
        return []


def _plus2_tickers(master: object) -> list[str]:
    try:
        attrs = getattr(master, "attributes", None)
        if attrs is None:
            return []
        items = attrs.items() if hasattr(attrs, "items") else []
        out: list[str] = []
        for k, v in items:
            try:
                if int(getattr(v, "leverage_multiple")) == 2:
                    out.append(str(k))
            except Exception:
                continue
        return out
    except Exception:
        return []


def _family_of(master: object, ticker: str) -> str | None:
    try:
        attrs = getattr(master, "attributes", None)
        if attrs is None:
            return None
        v = attrs.get(ticker) if hasattr(attrs, "get") else None
        if v is None:
            return None
        fam = getattr(v, "leverage_family_key", None)
        return str(fam) if fam is not None else None
    except Exception:
        return None


def select_attribution_windows(
    windows: pl.DataFrame,
    *,
    top_q: float = 0.95,
    near_miss_lo: float = 0.20,
    near_miss_hi: float = 0.50,
) -> pl.DataFrame:
    try:
        if windows is None or not isinstance(windows, pl.DataFrame) or windows.height == 0:
            return pl.DataFrame({"window_start": [], "window_end": [], "terminal_return": [], "giveback": []})
        if "terminal_return" not in windows.columns or "window_start" not in windows.columns:
            return windows.head(0)
        try:
            thresh = float(windows["terminal_return"].quantile(float(top_q)))
        except Exception:
            vals = sorted(float(v) for v in windows["terminal_return"].to_list() if v is not None)
            if not vals:
                return windows.head(0)
            import math

            q = min(max(float(top_q), 0.0), 1.0)
            pos = q * (len(vals) - 1)
            lo_i = int(math.floor(pos))
            hi_i = int(math.ceil(pos))
            thresh = vals[lo_i] if lo_i == hi_i else vals[lo_i] + (vals[hi_i] - vals[lo_i]) * (pos - lo_i)
        try:
            out = windows.filter(
                (pl.col("terminal_return") >= thresh)
                | ((pl.col("terminal_return") >= float(near_miss_lo)) & (pl.col("terminal_return") < float(near_miss_hi)))
            )
        except Exception:
            return windows.head(0)
        try:
            out = out.unique(subset=["window_start"], keep="first")
        except Exception:
            pass
        return out
    except Exception:
        try:
            return windows.head(0)
        except Exception:
            return pl.DataFrame({"window_start": [], "window_end": [], "terminal_return": [], "giveback": []})


def _zero_attribution(window_start: date, window_end: date, realized: float, giveback: float) -> WindowAttribution:
    try:
        r = float(realized)
    except Exception:
        r = 0.0
    try:
        gb = float(giveback)
    except Exception:
        gb = 0.0
    return WindowAttribution(
        window_start=window_start,
        window_end=window_end,
        realized_return=float(r),
        giveback=float(gb),
        best_family=None,
        best_family_return=0.0,
        actual_family=None,
        actual_family_bh_return=0.0,
        selection_loss=0.0,
        entry_timing_loss=0.0,
        exit_timing_loss=0.0,
        giveback_loss=max(0.0, float(gb)),
        dominant_bucket="UNKNOWN",
        r_actual=float(r),
        r_b=float(r),
        r_c=float(r),
        r_d=float(r),
        oracle_gap=0.0,
    )


def _actual_family_from_trades(
    trades: pl.DataFrame,
    master: object,
    window_start: date,
    window_end: date,
) -> str | None:
    try:
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        if (
            trades is not None
            and isinstance(trades, pl.DataFrame)
            and trades.height > 0
            and "decision_date" in trades.columns
            and "ticker" in trades.columns
            and "weight_after" in trades.columns
        ):
            order = 0
            for row in trades.iter_rows(named=True):
                try:
                    dd = row.get("decision_date")
                    wa = row.get("weight_after")
                    tk = row.get("ticker")
                    if dd is None or tk is None or wa is None:
                        continue
                    if not (window_start <= dd <= window_end):
                        continue
                    if float(wa) <= 1e-9:
                        continue
                    key = str(tk)
                    counts[key] = counts.get(key, 0) + 1
                    if key not in first_seen:
                        first_seen[key] = order
                    order += 1
                except Exception:
                    continue
        if not counts:
            return None
        top_ticker = sorted(counts.items(), key=lambda kv: (-kv[1], first_seen.get(kv[0], 0), kv[0]))[0][0]
        return _family_of(master, top_ticker)
    except Exception:
        return None


def _finite_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        fval = float(value)  # type: ignore[arg-type]
        import math

        if not math.isfinite(fval):
            return None
        return fval
    except Exception:
        return None
