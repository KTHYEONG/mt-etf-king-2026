# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from datetime import date

def locked_window_returns(
    daily_rets: Sequence[float], horizon: int, lock_level: float
) -> list[float]:
    try:
        h = int(horizon)
    except Exception:
        return []
    if h <= 0:
        return []
    if not daily_rets:
        return []
    try:
        ll = float(lock_level)
    except Exception:
        return []
    if not math.isfinite(ll) or ll <= 0:
        return []
    try:
        n = len(daily_rets)  # type: ignore[arg-type]
    except Exception:
        return []
    if n < h:
        return []
    out: list[float] = []
    for i in range(n - h + 1):
        equity = 1.0
        locked = False
        for k in range(h):
            if locked:
                continue
            try:
                r = float(daily_rets[i + k])  # type: ignore[index]
            except Exception:
                r = 0.0
            if not math.isfinite(r):
                r = 0.0
            # compound
            equity *= 1.0 + r
            if equity >= 1.0 + ll - 1e-12:
                locked = True
        out.append(float(equity - 1.0))
    return out


def championship_lock_returns(
    daily_rets: Sequence[float], horizon: int, arm: float, trail: float = 0.0
) -> list[float]:
    try:
        h = int(horizon)
    except Exception:
        return []
    if h <= 0:
        return []
    if not daily_rets:
        return []
    try:
        n = len(daily_rets)  # type: ignore[arg-type]
    except Exception:
        return []
    if n < h:
        return []
    try:
        ll = float(arm)
    except Exception:
        return []
    if not math.isfinite(ll) or ll <= 0:
        return []
    try:
        tr = float(trail)
    except Exception:
        tr = float("nan")
    if not math.isfinite(tr) or tr <= 0:
        return locked_window_returns(daily_rets, horizon, arm)
    # trail >0 hysteresis: floor-protected
    out: list[float] = []
    for i in range(n - h + 1):
        equity = 1.0
        peak = 1.0
        locked = False
        locked_equity = 1.0
        for k in range(h):
            if locked:
                continue
            try:
                r = float(daily_rets[i + k])  # type: ignore[index]
            except Exception:
                r = 0.0
            if not math.isfinite(r):
                r = 0.0
            equity *= 1.0 + r
            if equity > peak:
                peak = float(equity)
            stop = max(1.0 + ll, float(peak) - float(tr))
            if float(peak) > float(stop) and float(equity) < float(stop):
                equity = float(stop)
                locked = True
                peak = float(stop)
        out.append(float(equity - 1.0))
    return out


def house_money_ratchet_returns(
    daily_rets: Sequence[float], horizon: int, arm: float, lock_remaining: int = 5
) -> list[float]:
    try:
        h = int(horizon)
    except Exception:
        return []
    if h <= 0:
        return []
    if not daily_rets:
        return []
    try:
        n = len(daily_rets)  # type: ignore[arg-type]
    except Exception:
        return []
    if n < h:
        return []
    try:
        ll = float(arm)  # type: ignore[arg-type]
    except Exception:
        return []
    if not math.isfinite(ll) or ll <= 0:
        return []
    try:
        lf = float(lock_remaining)  # type: ignore[arg-type]
        if not math.isfinite(lf) or lf < 0:
            lr = 5
        else:
            lr = int(lf)
    except Exception:
        lr = 5
    out: list[float] = []
    for i in range(n - h + 1):
        equity = 1.0
        armed = False
        locked = False
        for k in range(h):
            if locked:
                break
            try:
                r = float(daily_rets[i + k])  # type: ignore[index]
            except Exception:
                r = 0.0
            if not math.isfinite(r):
                r = 0.0
            equity *= 1.0 + r
            if not armed and equity >= 1.0 + ll - 1e-12:
                armed = True
            if armed:
                if equity < 1.0 + ll - 1e-12:
                    equity = 1.0 + ll
                    locked = True
                else:
                    remaining_after = h - 1 - k
                    if remaining_after <= lr:
                        locked = True
        out.append(float(equity - 1.0))
    return out


def execution_faithful_late_lock_returns(
    daily_rets: Sequence[float], horizon: int, arm: float, lock_remaining: int
) -> list[float]:
    try:
        h = int(horizon)
    except Exception:
        return []
    if h <= 0:
        return []
    if not daily_rets:
        return []
    try:
        n = len(daily_rets)  # type: ignore[arg-type]
    except Exception:
        return []
    if n < h:
        return []
    try:
        ll = float(arm)  # type: ignore[arg-type]
    except Exception:
        return []
    if not math.isfinite(ll) or ll <= 0:
        return []
    try:
        lf = float(lock_remaining)  # type: ignore[arg-type]
        if not math.isfinite(lf) or lf < 0:
            lr = 5
        else:
            lr = int(lf)
    except Exception:
        lr = 5
    # use live predicate without floor invention
    from src.tournament.policy import house_money_should_cash

    out: list[float] = []
    for i in range(n - h + 1):
        equity = 1.0
        cashed = False
        cashed_equity = 1.0
        for k in range(h):
            if cashed:
                break
            try:
                r = float(daily_rets[i + k])  # type: ignore[index]
            except Exception:
                r = 0.0
            if not math.isfinite(r):
                r = 0.0
            equity *= 1.0 + r
            ret_from_start = float(equity - 1.0)
            remaining_after = h - 1 - k
            try:
                if house_money_should_cash(ret_from_start, remaining_after, ll, lr):
                    cashed = True
                    cashed_equity = float(equity)
                    break
            except Exception:
                continue
        if cashed:
            out.append(float(cashed_equity - 1.0))
        else:
            out.append(float(equity - 1.0))
    return out


def continuation_capture(
    unlocked: Sequence[float],
    freeze: Sequence[float],
    ratchet: Sequence[float],
    arm: float,
    *,
    eps: float = 0.10,
) -> float:
    try:
        af = float(arm)  # type: ignore[arg-type]
        if not math.isfinite(af):
            return 0.0
    except Exception:
        return 0.0
    try:
        ef = float(eps)  # type: ignore[arg-type]
        if not math.isfinite(ef):
            ef = 0.10
    except Exception:
        ef = 0.10
    threshold = af + ef
    n = min(len(unlocked), len(freeze), len(ratchet))
    if n == 0:
        return 0.0
    diffs: list[float] = []
    for idx in range(n):
        try:
            uv = float(unlocked[idx])  # type: ignore[index]
        except Exception:
            continue
        if not math.isfinite(uv):
            continue
        if uv > threshold:
            try:
                fv = float(freeze[idx])  # type: ignore[index]
                rv = float(ratchet[idx])  # type: ignore[index]
            except Exception:
                continue
            if not math.isfinite(fv) or not math.isfinite(rv):
                continue
            diffs.append(float(rv - fv))
    if not diffs:
        return 0.0
    return float(sum(diffs) / len(diffs))


def overlay_right_tail_stats(terminals: Sequence[float]) -> dict[str, float]:
    if not terminals:
        return {"q99": 0.0, "mean": 0.0, "p_gt_50": 0.0, "p_gt_60": 0.0, "p_gt_80": 0.0}
    try:
        vals = [float(x) for x in terminals]  # type: ignore[arg-type]
        vals = [v for v in vals if math.isfinite(v)]
    except Exception:
        return {"q99": 0.0, "mean": 0.0, "p_gt_50": 0.0, "p_gt_60": 0.0, "p_gt_80": 0.0}
    if not vals:
        return {"q99": 0.0, "mean": 0.0, "p_gt_50": 0.0, "p_gt_60": 0.0, "p_gt_80": 0.0}
    n = len(vals)
    mean_v = float(sum(vals) / n) if n else 0.0
    # q99 via linear interpolation pos = 0.99*(n-1)
    sorted_v = sorted(vals)
    if n == 1:
        q99 = float(sorted_v[0])
    else:
        pos = 0.99 * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            q99 = float(sorted_v[lo])
        else:
            frac = pos - lo
            q99 = float(sorted_v[lo]) * (1 - frac) + float(sorted_v[hi]) * frac
    def _p(th: float) -> float:
        cnt = sum(1 for v in vals if v > th)
        return float(cnt) / float(n) if n else 0.0
    return {
        "q99": float(q99),
        "mean": float(mean_v),
        "p_gt_50": float(_p(0.50)),
        "p_gt_60": float(_p(0.60)),
        "p_gt_80": float(_p(0.80)),
    }


def evaluate_p25_adoption_gates(
    q99_ratchet: float,
    q99_freeze: float,
    p_gt_60_ratchet: float,
    p_gt_60_freeze: float,
    p_gt_60_unlocked: float,
    continuation_capture: float,
    ruin: float,
    vehicle_rate: float,
    *,
    ruin_max: float = 0.05,
    min_vehicle_rate: float = 0.25,
) -> tuple[str, list[str]]:
    def _f(v: object) -> float:
        try:
            fv = float(v)  # type: ignore[arg-type]
            if not math.isfinite(fv):
                return 0.0
            return float(fv)
        except Exception:
            return 0.0
    q99_r = _f(q99_ratchet)
    q99_f = _f(q99_freeze)
    p60_r = _f(p_gt_60_ratchet)
    p60_f = _f(p_gt_60_freeze)
    p60_u = _f(p_gt_60_unlocked)
    cc = _f(continuation_capture)
    ru = _f(ruin)
    vr = _f(vehicle_rate)
    try:
        rm = float(ruin_max)  # type: ignore[arg-type]
        if not math.isfinite(rm):
            rm = 0.05
    except Exception:
        rm = 0.05
    try:
        mv = float(min_vehicle_rate)  # type: ignore[arg-type]
        if not math.isfinite(mv):
            mv = 0.25
    except Exception:
        mv = 0.25
    fails: list[str] = []
    if q99_r + 1e-12 < q99_f:
        fails.append("q99")
    if p60_u > 1e-12 and p60_r <= 1e-12:
        fails.append("right_tail_censored")
    if p60_r + 1e-12 < p60_f:
        fails.append("p_gt_60")
    if p60_u > 1e-12 and cc <= 1e-12:
        fails.append("continuation")
    if ru > rm + 1e-12:
        fails.append("ruin")
    if vr + 1e-12 < mv:
        fails.append("vehicle_activity")
    if not fails:
        return ("PASS", [])
    return ("FAIL", fails)


def oneshot_anchor_starts(
    sessions: Sequence[date], *, month: int = 9, day: int = 21, horizon: int = 36
) -> tuple[date, ...]:
    if not sessions or horizon <= 0:
        return ()
    try:
        m = int(month)  # type: ignore[arg-type]
        d = int(day)  # type: ignore[arg-type]
        h = int(horizon)  # type: ignore[arg-type]
    except Exception:
        return ()
    if m < 1 or m > 12 or d < 1 or d > 31 or h <= 0:
        return ()
    try:
        _ = date(2000, m, d)
    except Exception:
        return ()
    if not sessions:
        return ()
    try:
        sess_list = sorted(set(sessions))  # type: ignore[arg-type]
    except Exception:
        return ()
    if not sess_list:
        return ()
    idx_map = {s: i for i, s in enumerate(sess_list)}
    years = sorted({s.year for s in sess_list})
    out: list[date] = []
    for yr in years:
        try:
            anchor = date(yr, m, d)
        except Exception:
            continue
        candidate = None
        for s in sess_list:
            if s >= anchor and s.year == yr:
                candidate = s
                break
            if s >= anchor and s.year > yr:
                break
        if candidate is None:
            continue
        idx = idx_map.get(candidate)
        if idx is None:
            continue
        if idx + h <= len(sess_list):
            out.append(candidate)
    out = sorted(set(out))
    return tuple(out)


def oneshot_window_returns(
    daily_rets: Sequence[float],
    sessions: Sequence[date],
    starts: Sequence[date],
    horizon: int,
) -> tuple[tuple[int, date, float], ...]:
    if not daily_rets or not sessions or not starts or horizon <= 0:
        return ()
    try:
        h = int(horizon)  # type: ignore[arg-type]
    except Exception:
        return ()
    if h <= 0:
        return ()
    try:
        if len(daily_rets) != len(sessions):  # type: ignore[arg-type]
            return ()
    except Exception:
        return ()
    try:
        sess_list = list(sessions)  # type: ignore[arg-type]
        daily_list = list(daily_rets)  # type: ignore[arg-type]
        start_list = list(starts)  # type: ignore[arg-type]
    except Exception:
        return ()
    if not sess_list or not daily_list or not start_list:
        return ()
    idx_map2 = {s: i for i, s in enumerate(sess_list)}
    res: list[tuple[int, date, float]] = []
    for st in start_list:
        if st not in idx_map2:
            continue
        idx = idx_map2[st]
        if idx + h > len(sess_list):
            continue
        equity = 1.0
        for k in range(h):
            try:
                r = float(daily_list[idx + k])  # type: ignore[index]
            except Exception:
                r = 0.0
            if not math.isfinite(r):
                r = 0.0
            equity *= 1.0 + r
        ret = float(round(equity - 1.0, 12))
        res.append((int(st.year), st, float(ret)))
    return tuple(res)


def serialize_oneshot_rows(rows: Sequence[tuple[int, date, float]]) -> list[list[object]]:
    """JSON-safe oneshot rows: [[year, \"YYYY-MM-DD\", return], ...]."""
    out: list[list[object]] = []
    for row in rows:
        try:
            year, start, ret = row
            out.append([int(year), start.isoformat(), float(ret)])
        except Exception:
            continue
    return out


def evaluate_p24_adoption_gates(
    p_gt_30: float,
    b1_p_gt_30: float,
    p_gt_40: float,
    b1_p_gt_40: float,
    p_gt_50: float,
    b1_p_gt_50: float,
    cvar: float,
    b1_cvar: float,
    vehicle_rate: float,
    *,
    min_vehicle_rate: float = 0.25,
) -> tuple[str, list[str]]:
    def _cand(v: object) -> float:
        try:
            fv = float(v)  # type: ignore[arg-type]
        except Exception:
            return 0.0
        if not math.isfinite(fv):
            return 0.0
        return float(fv)

    def _b1(v: object) -> float:
        try:
            fv = float(v)  # type: ignore[arg-type]
        except Exception:
            return float("inf")
        if not math.isfinite(fv):
            return float("inf")
        return float(fv)

    p30 = _cand(p_gt_30)
    b30 = _b1(b1_p_gt_30)
    p40 = _cand(p_gt_40)
    b40 = _b1(b1_p_gt_40)
    p50 = _cand(p_gt_50)
    b50 = _b1(b1_p_gt_50)
    cv = _cand(cvar)
    bcv = _b1(b1_cvar)
    vr = _cand(vehicle_rate)
    try:
        mv = float(min_vehicle_rate)  # type: ignore[arg-type]
        if not math.isfinite(mv):
            mv = float("inf")
    except Exception:
        mv = float("inf")
    fails: list[str] = []
    if not (float(p30) >= float(b30) - 1e-12):
        fails.append("p_gt_30")
    if not (float(p40) >= float(b40) + 0.02 - 1e-12):
        fails.append("p_gt_40")
    if not (float(p50) >= float(b50) + 0.02 - 1e-12):
        fails.append("p_gt_50")
    if not (float(cv) >= float(bcv) - 0.05 - 1e-12):
        fails.append("cvar_05")
    if not (float(vr) >= float(mv) - 1e-12):
        fails.append("vehicle_activity")
    if not fails:
        return ("PASS", [])
    return ("FAIL", fails)
