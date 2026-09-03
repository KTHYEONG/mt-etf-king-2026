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
class TailMissReport:
    threshold: float
    n_windows: int
    n_near_miss: int
    label_counts: dict[str, int]
    top_windows: tuple[dict[str, object], ...]


def summarise_tail_miss_windows(
    windows: pl.DataFrame,
    candidates: pl.DataFrame,
    sessions: pl.DataFrame,
    *,
    threshold: float = 0.40,
    near_miss_lo: float = 0.20,
) -> TailMissReport:
    # fail-closed empty inputs
    try:
        if windows is None or not isinstance(windows, pl.DataFrame) or windows.height == 0:
            return TailMissReport(threshold=float(threshold), n_windows=0, n_near_miss=0, label_counts={}, top_windows=())
        if "terminal_return" not in windows.columns or "window_start" not in windows.columns or "window_end" not in windows.columns:
            # treat as zero near-miss but count windows
            n_w = int(windows.height)
            return TailMissReport(threshold=float(threshold), n_windows=n_w, n_near_miss=0, label_counts={}, top_windows=())
    except Exception:
        return TailMissReport(threshold=float(threshold), n_windows=0, n_near_miss=0, label_counts={}, top_windows=())
    n_windows = int(windows.height)
    # filter near-miss in [near_miss_lo, threshold)
    try:
        near = windows.filter(
            (pl.col("terminal_return") >= float(near_miss_lo)) & (pl.col("terminal_return") < float(threshold))
        )
    except Exception:
        # fallback iterative
        near_rows = []
        for r in windows.iter_rows(named=True):
            tr = r.get("terminal_return")
            try:
                if tr is not None and float(tr) >= float(near_miss_lo) and float(tr) < float(threshold):
                    near_rows.append(r)
            except Exception:
                continue
        if not near_rows:
            return TailMissReport(threshold=float(threshold), n_windows=n_windows, n_near_miss=0, label_counts={}, top_windows=())
        near = pl.DataFrame(near_rows)
    n_near_miss = int(near.height)
    if n_near_miss == 0:
        return TailMissReport(threshold=float(threshold), n_windows=n_windows, n_near_miss=0, label_counts={}, top_windows=())
    # build sessions map decision_date -> regime
    sess_map: dict[object, str] = {}
    try:
        if sessions is not None and isinstance(sessions, pl.DataFrame) and sessions.height > 0:
            if "decision_date" in sessions.columns and "regime" in sessions.columns:
                for row in sessions.iter_rows(named=True):
                    dd = row.get("decision_date")
                    rg = row.get("regime")
                    if dd is not None:
                        sess_map[dd] = str(rg) if rg is not None else ""
    except Exception:
        sess_map = {}
    # candidate rows
    cand_rows: list[dict[str, object]] = []
    try:
        if candidates is not None and isinstance(candidates, pl.DataFrame) and candidates.height > 0:
            cand_rows = list(candidates.iter_rows(named=True))  # type: ignore[assignment]
    except Exception:
        cand_rows = []
    label_counts: dict[str, int] = {}
    top_windows: list[dict[str, object]] = []
    risk_set = frozenset({"RISK_ON", "STRONG_RISK_ON"})
    for wrow in near.iter_rows(named=True):
        ws = wrow.get("window_start")
        we = wrow.get("window_end")
        tr = wrow.get("terminal_return")
        # join candidates where decision_date in [ws, we]
        matching: list[dict[str, object]] = []
        for crow in cand_rows:
            cd = crow.get("decision_date")
            if cd is None or ws is None or we is None:
                continue
            try:
                if cd >= ws and cd <= we:  # type: ignore[operator]
                    matching.append(crow)
            except Exception:
                continue
        if not matching:
            lab = "UNKNOWN"
            # still count
            label_counts[lab] = label_counts.get(lab, 0) + 1
            try:
                ws_str = ws.isoformat() if hasattr(ws, "isoformat") else str(ws)  # type: ignore[union-attr]
            except Exception:
                ws_str = str(ws)
            try:
                we_str = we.isoformat() if hasattr(we, "isoformat") else str(we)  # type: ignore[union-attr]
            except Exception:
                we_str = str(we)
            top_windows.append({"window_start": ws_str, "window_end": we_str, "terminal_return": float(tr) if tr is not None else 0.0, "label": lab})
            continue
        per_labels: list[str] = []
        for crow in matching:
            selected = crow.get("selected")
            # LOTTERY_INACTIVE
            lot_active = crow.get("lottery_active")
            if lot_active is not None and bool(lot_active) is False:
                per_labels.append("LOTTERY_INACTIVE")
                continue
            # MULT1_ONLY on selected
            is_selected = bool(selected) is True
            # need to handle selected may be None or bool-like
            try:
                is_selected = bool(selected) and selected is not False and selected is not None
                # but if selected is True exactly
                if selected is True:
                    is_selected = True
                elif selected is False:
                    is_selected = False
                else:
                    # polars bool may be bool
                    is_selected = bool(selected)
            except Exception:
                is_selected = False
            # If column missing, is_selected stays False
            mult = crow.get("multiple")
            if is_selected:
                try:
                    if mult is not None and int(mult) == 1:  # type: ignore[arg-type]
                        per_labels.append("MULT1_ONLY")
                        continue
                except Exception:
                    pass
            # LOW_GROSS weight_fill <0.5 on selected
            wf = crow.get("weight_fill")
            if wf is None:
                wf = crow.get("weight_filled")
            if wf is None:
                wf = crow.get("weight_after_capacity")
            if is_selected and wf is not None:
                try:
                    if float(wf) < 0.5:  # type: ignore[arg-type]
                        per_labels.append("LOW_GROSS")
                        continue
                except Exception:
                    pass
            # REGIME_OFF regime not in RISK_ON/STRONG_RISK_ON
            cd = crow.get("decision_date")
            reg = sess_map.get(cd, None)
            if reg is None:
                # try regime from candidate row itself
                reg = crow.get("regime")  # type: ignore[assignment]
            reg_str = str(reg) if reg is not None else ""
            if reg_str not in risk_set:
                per_labels.append("REGIME_OFF")
                continue
            per_labels.append("UNKNOWN")
        # majority vote
        from collections import Counter

        cnt = Counter(per_labels)
        if cnt:
            max_c = max(cnt.values())
            priority = ["LOTTERY_INACTIVE", "MULT1_ONLY", "LOW_GROSS", "REGIME_OFF", "UNKNOWN"]
            best = None
            for p in priority:
                if cnt.get(p, 0) == max_c:
                    best = p
                    break
            if best is None:
                best = cnt.most_common(1)[0][0]
            lab = str(best)
        else:
            lab = "UNKNOWN"
        label_counts[lab] = label_counts.get(lab, 0) + 1
        try:
            ws_str = ws.isoformat() if hasattr(ws, "isoformat") else str(ws)  # type: ignore[union-attr]
        except Exception:
            ws_str = str(ws)
        try:
            we_str = we.isoformat() if hasattr(we, "isoformat") else str(we)  # type: ignore[union-attr]
        except Exception:
            we_str = str(we)
        top_windows.append({"window_start": ws_str, "window_end": we_str, "terminal_return": float(tr) if tr is not None else 0.0, "label": lab})
    return TailMissReport(
        threshold=float(threshold),
        n_windows=n_windows,
        n_near_miss=n_near_miss,
        label_counts=dict(label_counts),
        top_windows=tuple(top_windows),
    )


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
def write_tail_miss_report(dest: Path, report: TailMissReport) -> str:
    _ = "write_tail_attribution_report"
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "tail_miss_report.json"
    data = {
        "threshold": float(report.threshold),
        "n_windows": int(report.n_windows),
        "n_near_miss": int(report.n_near_miss),
        "label_counts": dict(report.label_counts),
        "top_windows": list(report.top_windows),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(out_path)
