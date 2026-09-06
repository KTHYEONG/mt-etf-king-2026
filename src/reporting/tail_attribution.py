# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl

from src.reporting.tail_windows import (
    TailAttributionSummary,
    WindowAttribution,
    _actual_family_from_trades,
    _family_of,
    _finite_or_none,
    _zero_attribution,
    next_open_path_return,
    oracle_peak_path_return,
    pit_plus2_tickers,
    select_attribution_windows,
)


def attribute_window(
    *,
    window_start: date,
    window_end: date,
    realized_return: float,
    giveback: float,
    trades: pl.DataFrame,
    panel: pl.DataFrame,
    master: object,
    sessions: Sequence[date],
    universe: object | None = None,
    filters: object | None = None,
) -> WindowAttribution:
    try:
        realized = float(realized_return)
    except Exception:
        realized = 0.0
    try:
        gb = float(giveback)
    except Exception:
        gb = 0.0
    giveback_loss = max(0.0, gb)
    try:
        candidates = pit_plus2_tickers(master, window_start=window_start, universe=universe, filters=filters)
        actual_family = _actual_family_from_trades(trades, master, window_start, window_end)
        # D = PIT-eligible best +2x oracle peak (entry+exit)
        oracle_ticker: str | None = None
        d_raw: float | None = None
        for t in candidates:
            try:
                r = oracle_peak_path_return(panel, t, window_start, window_end, sessions)
            except Exception:
                r = None
            fval = _finite_or_none(r)
            if fval is None:
                continue
            if d_raw is None or fval > d_raw:
                d_raw = fval
                oracle_ticker = t
        best_fam = _family_of(master, oracle_ticker) if oracle_ticker is not None else None
        if actual_family is None:
            d_eff = float(d_raw) if d_raw is not None else float(realized)
            return WindowAttribution(
                window_start=window_start,
                window_end=window_end,
                realized_return=realized,
                giveback=gb,
                best_family=best_fam,
                best_family_return=float(d_raw) if d_raw is not None else 0.0,
                actual_family=None,
                actual_family_bh_return=0.0,
                selection_loss=0.0,
                entry_timing_loss=0.0,
                exit_timing_loss=0.0,
                giveback_loss=giveback_loss,
                dominant_bucket="UNKNOWN",
                r_actual=float(realized),
                r_b=float(realized),
                r_c=float(realized),
                r_d=float(d_eff),
                oracle_gap=max(0.0, float(d_eff) - float(realized)),
            )
        # B = actual-family best with oracle entry, hold-to-end
        members = [t for t in candidates if _family_of(master, t) == actual_family]
        b_ticker: str | None = None
        b_raw: float | None = None
        for t in members:
            try:
                r = next_open_path_return(panel, t, window_start, window_end, sessions)
            except Exception:
                r = None
            fval = _finite_or_none(r)
            if fval is None:
                continue
            if b_raw is None or fval > b_raw:
                b_raw = fval
                b_ticker = t
        # C = same vehicle with oracle exit (peak mark)
        c_raw: float | None = None
        if b_ticker is not None:
            try:
                c_raw = oracle_peak_path_return(panel, b_ticker, window_start, window_end, sessions)
            except Exception:
                c_raw = None
            c_raw = _finite_or_none(c_raw)
        b = float(b_raw) if b_raw is not None else float(realized)
        c = float(c_raw) if c_raw is not None else float(b)
        d = float(d_raw) if d_raw is not None else float(c)
        entry_timing_loss = max(0.0, float(b) - float(realized))
        exit_timing_loss = max(0.0, float(c) - float(b))
        selection_loss = max(0.0, float(d) - float(c))
        if oracle_ticker is None:
            dom = "UNKNOWN"
        elif selection_loss == 0.0 and entry_timing_loss == 0.0 and exit_timing_loss == 0.0:
            dom = "NONE"
        else:
            order = ["selection", "entry_timing", "exit_timing"]
            losses = {"selection": selection_loss, "entry_timing": entry_timing_loss, "exit_timing": exit_timing_loss}
            dom = order[0]
            best_v = losses[order[0]]
            for k in order[1:]:
                if losses[k] > best_v:
                    best_v = losses[k]
                    dom = k
        return WindowAttribution(
            window_start=window_start,
            window_end=window_end,
            realized_return=realized,
            giveback=gb,
            best_family=best_fam,
            best_family_return=float(d_raw) if d_raw is not None else 0.0,
            actual_family=actual_family,
            actual_family_bh_return=float(b_raw) if b_raw is not None else 0.0,
            selection_loss=float(selection_loss),
            entry_timing_loss=float(entry_timing_loss),
            exit_timing_loss=float(exit_timing_loss),
            giveback_loss=float(giveback_loss),
            dominant_bucket=str(dom),
            r_actual=float(realized),
            r_b=float(b),
            r_c=float(c),
            r_d=float(d),
            oracle_gap=max(0.0, float(d) - float(realized)),
        )
    except Exception:
        try:
            return _zero_attribution(window_start, window_end, realized, gb)
        except Exception:
            return WindowAttribution(
                window_start=window_start,
                window_end=window_end,
                realized_return=0.0,
                giveback=0.0,
                best_family=None,
                best_family_return=0.0,
                actual_family=None,
                actual_family_bh_return=0.0,
                selection_loss=0.0,
                entry_timing_loss=0.0,
                exit_timing_loss=0.0,
                giveback_loss=0.0,
                dominant_bucket="UNKNOWN",
                r_actual=0.0,
                r_b=0.0,
                r_c=0.0,
                r_d=0.0,
                oracle_gap=0.0,
            )


def _median(values: list[float]) -> float:
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def _trimmed_mean(values: list[float], frac: float = 0.10) -> float:
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n < 10:
        return float(sum(vals) / n)
    k = int(n * frac // 1)
    if k <= 0:
        return float(sum(vals) / n)
    rest = vals[k : n - k] if n - 2 * k > 0 else vals
    if not rest:
        return float(sum(vals) / n)
    return float(sum(rest) / len(rest))


def _quantile(values: list[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return 0.0
    qq = min(max(float(q), 0.0), 1.0)
    pos = qq * (n - 1)
    import math

    lo_i = int(math.floor(pos))
    hi_i = int(math.ceil(pos))
    if lo_i == hi_i:
        return float(vals[lo_i])
    return float(vals[lo_i] + (vals[hi_i] - vals[lo_i]) * (pos - lo_i))


def summarise_tail_attribution(
    *,
    windows: pl.DataFrame,
    trades: pl.DataFrame,
    panel: pl.DataFrame,
    master: object,
    sessions: Sequence[date],
    top_q: float = 0.95,
    near_miss_lo: float = 0.20,
    near_miss_hi: float = 0.50,
    universe: object | None = None,
    filters: object | None = None,
) -> TailAttributionSummary:
    try:
        n_total = int(windows.height) if windows is not None and isinstance(windows, pl.DataFrame) else 0
    except Exception:
        n_total = 0
    try:
        selected = select_attribution_windows(windows, top_q=float(top_q), near_miss_lo=float(near_miss_lo), near_miss_hi=float(near_miss_hi))
    except Exception:
        selected = windows.head(0) if isinstance(windows, pl.DataFrame) else pl.DataFrame()
    attrs: list[WindowAttribution] = []
    try:
        if selected is not None and isinstance(selected, pl.DataFrame) and selected.height > 0:
            for row in selected.iter_rows(named=True):
                try:
                    ws = row.get("window_start")
                    we = row.get("window_end")
                    tr = row.get("terminal_return", 0.0)
                    gb = row.get("giveback", 0.0)
                    if ws is None or we is None:
                        continue
                    attrs.append(
                        attribute_window(
                            window_start=ws,
                            window_end=we,
                            realized_return=float(tr) if tr is not None else 0.0,
                            giveback=float(gb) if gb is not None else 0.0,
                            trades=trades,
                            panel=panel,
                            master=master,
                            sessions=sessions,
                            universe=universe,
                            filters=filters,
                        )
                    )
                except Exception:
                    continue
    except Exception:
        attrs = []
    n_an = len(attrs)
    if n_an == 0:
        return TailAttributionSummary(
            n_windows_total=n_total,
            n_analyzed=0,
            mean_selection_loss=0.0,
            mean_entry_timing_loss=0.0,
            mean_exit_timing_loss=0.0,
            mean_giveback_loss=0.0,
            selection_dominates_timing=False,
            primary_gap="INSUFFICIENT",
            bucket_counts={},
            windows=(),
            era_means={},
        )
    sel = [float(a.selection_loss) for a in attrs]
    ent = [float(a.entry_timing_loss) for a in attrs]
    ext = [float(a.exit_timing_loss) for a in attrs]
    m_sel = sum(sel) / n_an
    m_entry = sum(ent) / n_an
    m_exit = sum(ext) / n_an
    try:
        m_gb = sum(float(a.giveback_loss) for a in attrs) / n_an
    except Exception:
        m_gb = 0.0
    dom = bool(m_sel > (m_entry + m_exit))
    med_sel = _median(sel)
    med_ent = _median(ent)
    med_ext = _median(ext)
    if med_sel == 0.0 and med_ent == 0.0 and med_ext == 0.0:
        gap = "NONE"
    else:
        order = ["selection", "entry_timing", "exit_timing"]
        meds = {"selection": med_sel, "entry_timing": med_ent, "exit_timing": med_ext}
        gap = order[0]
        best_v = meds[order[0]]
        for k in order[1:]:
            if meds[k] > best_v:
                best_v = meds[k]
                gap = k
    denom = sum(sel) + sum(ent) + sum(ext)
    if denom > 0:
        share_sel = sum(sel) / denom
        share_ent = sum(ent) / denom
        share_ext = sum(ext) / denom
    else:
        share_sel = 0.0
        share_ent = 0.0
        share_ext = 0.0
    era: dict[str, dict[str, float]] = {}
    try:
        by_year: dict[str, list[WindowAttribution]] = {}
        for a in attrs:
            try:
                year = str(a.window_start.year)
            except Exception:
                year = "unknown"
            by_year.setdefault(year, []).append(a)
        for year, group in by_year.items():
            gn = len(group)
            era[year] = {
                "selection": sum(float(x.selection_loss) for x in group) / gn if gn else 0.0,
                "entry_timing": sum(float(x.entry_timing_loss) for x in group) / gn if gn else 0.0,
                "exit_timing": sum(float(x.exit_timing_loss) for x in group) / gn if gn else 0.0,
            }
    except Exception:
        era = {}
    counts: dict[str, int] = {}
    for a in attrs:
        counts[a.dominant_bucket] = counts.get(a.dominant_bucket, 0) + 1
    return TailAttributionSummary(
        n_windows_total=n_total,
        n_analyzed=n_an,
        mean_selection_loss=float(m_sel),
        mean_entry_timing_loss=float(m_entry),
        mean_exit_timing_loss=float(m_exit),
        mean_giveback_loss=float(m_gb),
        selection_dominates_timing=dom,
        primary_gap=gap,
        bucket_counts=dict(counts),
        windows=tuple(attrs),
        median_selection_loss=float(med_sel),
        median_entry_timing_loss=float(med_ent),
        median_exit_timing_loss=float(med_ext),
        trimmed_mean_selection_loss=float(_trimmed_mean(sel)),
        trimmed_mean_entry_timing_loss=float(_trimmed_mean(ent)),
        trimmed_mean_exit_timing_loss=float(_trimmed_mean(ext)),
        q75_selection_loss=float(_quantile(sel, 0.75)),
        q75_entry_timing_loss=float(_quantile(ent, 0.75)),
        q75_exit_timing_loss=float(_quantile(ext, 0.75)),
        q90_selection_loss=float(_quantile(sel, 0.90)),
        q90_entry_timing_loss=float(_quantile(ent, 0.90)),
        q90_exit_timing_loss=float(_quantile(ext, 0.90)),
        share_selection=float(share_sel),
        share_entry_timing=float(share_ent),
        share_exit_timing=float(share_ext),
        era_means=dict(era),
    )


def write_tail_attribution_report(dest: Path, summary: TailAttributionSummary) -> str:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "tail_attribution_report.json"

    def _rec(a: WindowAttribution) -> dict[str, object]:
        def _iso(v: object) -> str:
            try:
                return v.isoformat()  # type: ignore[union-attr]
            except Exception:
                return str(v)

        return {
            "window_start": _iso(a.window_start),
            "window_end": _iso(a.window_end),
            "realized_return": float(a.realized_return),
            "giveback": float(a.giveback),
            "best_family": a.best_family,
            "best_family_return": float(a.best_family_return),
            "actual_family": a.actual_family,
            "actual_family_bh_return": float(a.actual_family_bh_return),
            "selection_loss": float(a.selection_loss),
            "entry_timing_loss": float(a.entry_timing_loss),
            "exit_timing_loss": float(a.exit_timing_loss),
            "giveback_loss": float(a.giveback_loss),
            "dominant_bucket": str(a.dominant_bucket),
            "r_actual": float(a.r_actual),
            "r_b": float(a.r_b),
            "r_c": float(a.r_c),
            "r_d": float(a.r_d),
            "oracle_gap": float(a.oracle_gap),
        }

    data = {
        "n_windows_total": int(summary.n_windows_total),
        "n_analyzed": int(summary.n_analyzed),
        "mean_selection_loss": float(summary.mean_selection_loss),
        "mean_entry_timing_loss": float(summary.mean_entry_timing_loss),
        "mean_exit_timing_loss": float(summary.mean_exit_timing_loss),
        "mean_giveback_loss": float(summary.mean_giveback_loss),
        "selection_dominates_timing": bool(summary.selection_dominates_timing),
        "primary_gap": str(summary.primary_gap),
        "bucket_counts": dict(summary.bucket_counts),
        "median_selection_loss": float(summary.median_selection_loss),
        "median_entry_timing_loss": float(summary.median_entry_timing_loss),
        "median_exit_timing_loss": float(summary.median_exit_timing_loss),
        "trimmed_mean_selection_loss": float(summary.trimmed_mean_selection_loss),
        "trimmed_mean_entry_timing_loss": float(summary.trimmed_mean_entry_timing_loss),
        "trimmed_mean_exit_timing_loss": float(summary.trimmed_mean_exit_timing_loss),
        "q75_selection_loss": float(summary.q75_selection_loss),
        "q75_entry_timing_loss": float(summary.q75_entry_timing_loss),
        "q75_exit_timing_loss": float(summary.q75_exit_timing_loss),
        "q90_selection_loss": float(summary.q90_selection_loss),
        "q90_entry_timing_loss": float(summary.q90_entry_timing_loss),
        "q90_exit_timing_loss": float(summary.q90_exit_timing_loss),
        "share_selection": float(summary.share_selection),
        "share_entry_timing": float(summary.share_entry_timing),
        "share_exit_timing": float(summary.share_exit_timing),
        "era_means": dict(summary.era_means),
        "windows": [_rec(a) for a in summary.windows],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(out_path)
