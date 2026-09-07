# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

import math

import polars as pl


def capacity_frontier_scores(
    snapshot: pl.DataFrame,
    *,
    capital: float,
    remaining_sessions: int,
    participation: float,
    cost_bps: float,
) -> dict[str, float]:
    required = ("ticker", "mom_60", "adv20", "leverage_multiple", "confidence", "eligible")
    if any(c not in snapshot.columns for c in required) or snapshot.height != snapshot.select("ticker").unique().height:
        raise ValueError("invalid snapshot schema or duplicate tickers")
    if not isinstance(remaining_sessions, int) or isinstance(remaining_sessions, bool) or not (0 <= remaining_sessions <= 36):
        raise ValueError("remaining_sessions must be integer 0..36")
    if not isinstance(capital, (int, float)) or not math.isfinite(float(capital)) or float(capital) <= 0:
        raise ValueError("capital must be positive finite")
    if not isinstance(participation, (int, float)) or not math.isfinite(float(participation)) or float(participation) <= 0:
        raise ValueError("participation must be positive finite")
    if not isinstance(cost_bps, (int, float)) or not math.isfinite(float(cost_bps)) or float(cost_bps) < 0:
        raise ValueError("cost_bps must be nonnegative finite")
    if remaining_sessions == 0:
        return {}
    r = remaining_sessions
    out: dict[str, float] = {}
    for row in snapshot.iter_rows(named=True):
        t = str(row["ticker"])
        mom = row["mom_60"]
        adv = row["adv20"]
        mult = row["leverage_multiple"]
        conf = str(row["confidence"])
        elig = bool(row["eligible"])
        if not elig:
            continue
        if not isinstance(mom, (int, float)) or not math.isfinite(float(mom)) or float(mom) <= 0:
            continue
        if not isinstance(adv, (int, float)) or not math.isfinite(float(adv)) or float(adv) <= 0:
            continue
        if not isinstance(mult, (int, float)) or not math.isfinite(float(mult)) or float(mult) != float(int(float(mult))):
            continue
        m = int(mult)
        if m not in (-2, -1, 1, 2):
            continue
        if m in (-2, -1, 2) and conf != "HIGH":
            continue
        cap = float(adv) * float(participation) / float(capital)
        tgt = min(0.80, 1.60 / abs(m), 0.95)
        avg = sum(min(tgt, d * cap) for d in range(1, r + 1)) / r
        score = max(0.0, float(mom) * (r / 60.0) * avg - 2.0 * (float(cost_bps) / 10000.0) * min(tgt, r * cap))
        if score > 0:
            out[t] = float(score)
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
