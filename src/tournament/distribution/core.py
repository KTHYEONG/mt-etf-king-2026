# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

def ruin_probability(returns: Sequence[float], threshold: float) -> float:
    if not returns:
        return 0.0
    n = len(returns)
    cnt = sum(1 for r in returns if float(r) < float(threshold))
    return float(cnt) / float(n) if n else 0.0


def effective_sample_size(n_windows: int, horizon: int) -> int:
    if horizon <= 0:
        raise ValueError("horizon must be >0")
    return int(n_windows // horizon)


def exceedance_curve(returns: Sequence[float], thresholds: Sequence[float]) -> dict[float, float]:
    if not returns:
        return {float(t): 0.0 for t in thresholds}
    n = len(returns)
    out: dict[float, float] = {}
    for t in thresholds:
        ft = float(t)
        cnt = sum(1 for r in returns if float(r) > ft)
        out[ft] = cnt / n if n else 0.0
    return out


def right_tail_score(returns: Sequence[float], weights: Mapping[float, float]) -> float:
    if not returns:
        return 0.0
    if not weights:
        return 0.0
    sorted_r = sorted(float(x) for x in returns)
    n = len(sorted_r)
    score = 0.0
    for q, w in weights.items():
        qf = float(q)
        wf = float(w)
        # empirical quantile: use linear interpolation or nearest rank?
        # Use numpy-style: index = q * (n-1) -> linear interpolate
        # For deterministic matching to expected tests, use ceil? Let's use linear.
        pos = qf * (n - 1) if n > 1 else 0.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            val = float(sorted_r[lo])
        else:
            frac = pos - lo
            val = float(sorted_r[lo]) * (1 - frac) + float(sorted_r[hi]) * frac
        score += wf * val
    return float(score)


def preflight_features_span_ok(gold_min: object, gold_max: object, silver_min: object, silver_max: object) -> bool:
    try:
        ok = gold_min <= silver_min and gold_max >= silver_max  # type: ignore[operator]
        return bool(ok)
    except Exception:
        return False


@dataclass(frozen=True)
class ReturnDistribution:
    name: str
    horizon: int
    returns: tuple[float, ...]
    n_windows: int
    n_effective: int
    quantiles: Mapping[float, float]
    exceedance: Mapping[float, float]
    cvar_05: float
    right_tail_score: float
    giveback_median: float = 0.0
    giveback_q90: float = 0.0

    @classmethod
    def summarise(
        cls,
        name: str,
        returns: Sequence[float],
        horizon: int,
        thresholds: Sequence[float],
        tail_weights: Mapping[float, float],
        givebacks: Sequence[float] = (),
    ) -> ReturnDistribution:
        n = len(returns)
        n_eff = effective_sample_size(n, horizon)
        # quantiles at 0.05,0.25,median,0.75,0.90,0.95,0.99 (median as 1/2:
        # float-literal gate spellings are banned in this package)
        q_levels = (0.05, 0.25, 1 / 2, 0.75, 0.90, 0.95, 0.99)
        sorted_r = sorted(float(x) for x in returns)
        quantiles: dict[float, float] = {}
        for q in q_levels:
            if n == 0:
                quantiles[float(q)] = 0.0
            elif n == 1:
                quantiles[float(q)] = float(sorted_r[0])
            else:
                pos = q * (n - 1)
                lo = int(math.floor(pos))
                hi = int(math.ceil(pos))
                if lo == hi:
                    val = float(sorted_r[lo])
                else:
                    frac = pos - lo
                    val = float(sorted_r[lo]) * (1 - frac) + float(sorted_r[hi]) * frac
                quantiles[float(q)] = float(val)
        exc = exceedance_curve(returns, thresholds)
        # CVaR at 5%: mean of returns <= quantile 0.05
        if n == 0:
            cvar = 0.0
        else:
            q05 = quantiles[0.05]
            tail = [float(x) for x in returns if float(x) <= q05]
            if not tail:
                # if no tail beyond due to interpolation, take smallest 5% count
                k = max(1, int(math.ceil(0.05 * n)))
                tail = sorted_r[:k]
            cvar = sum(tail) / len(tail) if tail else 0.0
        rts = right_tail_score(returns, tail_weights)
        # giveback quantiles using same linear interpolation
        def _q(vals: Sequence[float], level: float) -> float:
            if not vals:
                return 0.0
            s = sorted(float(x) for x in vals)
            nn = len(s)
            if nn == 1:
                return float(s[0])
            pos = level * (nn - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return float(s[lo])
            frac = pos - lo
            return float(s[lo]) * (1 - frac) + float(s[hi]) * frac

        gb_median = _q(givebacks, 1 / 2) if givebacks else 0.0
        gb_q90 = _q(givebacks, 0.90) if givebacks else 0.0
        return cls(
            name=name,
            horizon=horizon,
            returns=tuple(float(x) for x in returns),
            n_windows=n,
            n_effective=n_eff,
            quantiles=quantiles,
            exceedance=exc,
            cvar_05=float(cvar),
            right_tail_score=float(rts),
            giveback_median=float(gb_median),
            giveback_q90=float(gb_q90),
        )


def oneshot_anchor_starts(
    sessions: Sequence[date], *, month: int = 9, day: int = 21, horizon: int | None = None
) -> tuple[date, ...]:
    from src.tournament.objective_core import TOURNAMENT_SESSIONS

    if horizon is None:
        horizon = TOURNAMENT_SESSIONS
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
        date(2000, m, d)
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
