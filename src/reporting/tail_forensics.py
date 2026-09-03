# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
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
    return WindowAttribution(
        window_start=window_start,
        window_end=window_end,
        realized_return=float(realized),
        giveback=float(giveback),
        best_family=None,
        best_family_return=0.0,
        actual_family=None,
        actual_family_bh_return=0.0,
        selection_loss=0.0,
        entry_timing_loss=0.0,
        exit_timing_loss=0.0,
        giveback_loss=max(0.0, float(giveback)),
        dominant_bucket="UNKNOWN",
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
        plus2 = _plus2_tickers(master)
        best_ticker: str | None = None
        best_ret = 0.0
        best_fam: str | None = None
        for t in plus2:
            try:
                r = compound_close_return(panel, t, window_start, window_end)
            except Exception:
                r = None
            if r is None:
                continue
            try:
                import math

                if not math.isfinite(float(r)):
                    continue
            except Exception:
                continue
            if best_ticker is None or float(r) > best_ret:
                best_ticker = t
                best_ret = float(r)
                best_fam = _family_of(master, t)
        if best_ticker is None:
            best_fam = None
            best_ret = 0.0
        # Actual family from trades
        actual_family: str | None = None
        try:
            counts: dict[str, int] = {}
            first_seen: dict[str, int] = {}
            if trades is not None and isinstance(trades, pl.DataFrame) and trades.height > 0 and "decision_date" in trades.columns and "ticker" in trades.columns and "weight_after" in trades.columns:
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
            if counts:
                top_ticker = sorted(counts.items(), key=lambda kv: (-kv[1], first_seen.get(kv[0], 0), kv[0]))[0][0]
                actual_family = _family_of(master, top_ticker)
                if actual_family is None:
                    # ticker missing attributes -> treat as no actual
                    actual_family = None
                    counts = {}
            if actual_family is None:
                return WindowAttribution(
                    window_start=window_start,
                    window_end=window_end,
                    realized_return=realized,
                    giveback=gb,
                    best_family=best_fam,
                    best_family_return=float(best_ret),
                    actual_family=None,
                    actual_family_bh_return=0.0,
                    selection_loss=0.0,
                    entry_timing_loss=0.0,
                    exit_timing_loss=0.0,
                    giveback_loss=giveback_loss,
                    dominant_bucket="UNKNOWN",
                )
        except Exception:
            return WindowAttribution(
                window_start=window_start,
                window_end=window_end,
                realized_return=realized,
                giveback=gb,
                best_family=None,
                best_family_return=0.0,
                actual_family=None,
                actual_family_bh_return=0.0,
                selection_loss=0.0,
                entry_timing_loss=0.0,
                exit_timing_loss=0.0,
                giveback_loss=giveback_loss,
                dominant_bucket="UNKNOWN",
            )
        # actual family BH return: max compound among +2 members of actual family
        members = [t for t in plus2 if _family_of(master, t) == actual_family]
        actual_bh = 0.0
        best_actual_ticker: str | None = None
        for t in members:
            try:
                r = compound_close_return(panel, t, window_start, window_end)
            except Exception:
                r = None
            if r is None:
                continue
            if best_actual_ticker is None or float(r) > actual_bh:
                best_actual_ticker = t
                actual_bh = float(r)
        selection_loss = max(0.0, float(best_ret) - float(actual_bh))
        # entry timing
        first_entry: date | None = None
        try:
            if trades is not None and isinstance(trades, pl.DataFrame) and trades.height > 0:
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
                        if _family_of(master, str(tk)) != actual_family:
                            continue
                        if first_entry is None or dd < first_entry:
                            first_entry = dd
                    except Exception:
                        continue
        except Exception:
            first_entry = None
        entry_bh = 0.0
        entry_ticker: str | None = None
        if first_entry is not None:
            cand_best = 0.0
            cand_ticker: str | None = None
            found = False
            for t in members:
                try:
                    r = compound_close_return(panel, t, first_entry, window_end)
                except Exception:
                    r = None
                if r is None:
                    continue
                if not found or float(r) > cand_best:
                    cand_best = float(r)
                    cand_ticker = t
                    found = True
            if found:
                entry_bh = float(cand_best)
                entry_ticker = cand_ticker
            else:
                entry_bh = 0.0
        entry_timing_loss = max(0.0, float(actual_bh) - float(entry_bh)) if first_entry is not None else 0.0
        # exit timing: max compound(first_entry -> e) for entry vehicle over sessions in [first_entry, window_end]
        exit_bh = float(entry_bh)
        if first_entry is not None and entry_ticker is not None:
            try:
                sess_list = [s for s in list(sessions) if first_entry <= s <= window_end]
            except Exception:
                sess_list = []
            peak = float(entry_bh)
            for e in sess_list:
                try:
                    r = compound_close_return(panel, entry_ticker, first_entry, e)
                except Exception:
                    r = None
                if r is None:
                    continue
                if float(r) > peak:
                    peak = float(r)
            exit_bh = peak
        exit_timing_loss = max(0.0, float(exit_bh) - float(entry_bh)) if first_entry is not None else 0.0
        losses = {"selection": selection_loss, "entry_timing": entry_timing_loss, "exit_timing": exit_timing_loss, "giveback": giveback_loss}
        if all(v == 0.0 for v in losses.values()):
            dom = "NONE"
        else:
            order = ["selection", "entry_timing", "exit_timing", "giveback"]
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
            best_family_return=float(best_ret),
            actual_family=actual_family,
            actual_family_bh_return=float(actual_bh),
            selection_loss=float(selection_loss),
            entry_timing_loss=float(entry_timing_loss),
            exit_timing_loss=float(exit_timing_loss),
            giveback_loss=float(giveback_loss),
            dominant_bucket=str(dom),
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
            )


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
        )
    m_sel = sum(a.selection_loss for a in attrs) / n_an
    m_entry = sum(a.entry_timing_loss for a in attrs) / n_an
    m_exit = sum(a.exit_timing_loss for a in attrs) / n_an
    m_gb = sum(a.giveback_loss for a in attrs) / n_an
    dom = bool(m_sel > (m_entry + m_exit))
    if dom:
        gap = "selection"
    elif (m_entry + m_exit) >= m_gb:
        gap = "timing"
    else:
        gap = "giveback"
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
