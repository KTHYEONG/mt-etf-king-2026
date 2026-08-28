from __future__ import annotations

import math
from collections.abc import Sequence


def max_drawdown(equity: Sequence[float]) -> float:
    if not equity:
        return 0.0
    peak = float(equity[0])
    mdd = 0.0
    for v in equity:
        fv = float(v)
        if fv > peak:
            peak = fv
        if peak == 0:
            continue
        dd = (fv - peak) / peak
        if dd < mdd:
            mdd = dd
    return float(mdd)


def compound_returns(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    # Use log1p/expm1 for numerical accuracy
    s = 0.0
    for r in returns:
        s += math.log1p(float(r))
    return float(math.expm1(s))


def window_returns(returns: Sequence[float], horizon: int) -> list[float]:
    n = len(returns)
    if horizon <= 0:
        raise ValueError("horizon must be >0")
    if n < horizon:
        return []
    # Compute cumulative log returns
    cum = [0.0] * (n + 1)
    for i, r in enumerate(returns):
        cum[i + 1] = cum[i] + math.log1p(float(r))
    out: list[float] = []
    for i in range(n - horizon + 1):
        window_log = cum[i + horizon] - cum[i]
        out.append(float(math.expm1(window_log)))
    return out
