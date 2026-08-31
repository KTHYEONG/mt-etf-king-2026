# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import json
from dataclasses import dataclass
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


def write_tail_miss_report(dest: Path, report: TailMissReport) -> str:
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
