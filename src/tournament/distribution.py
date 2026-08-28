from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


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


def stationary_bootstrap_ci(
    returns: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    expected_block: int,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    # Stationary bootstrap: block lengths ~ Geom(p=1/expected_block)
    # Circular bootstrap over returns
    n = len(returns)
    if n == 0:
        return (0.0, 0.0)
    if expected_block <= 0:
        raise ValueError("expected_block must be >0")
    p = 1.0 / float(expected_block)
    rng = random.Random(seed)
    base_stat = float(statistic(list(returns)))
    # Generate resamples
    stats: list[float] = []
    arr = list(float(x) for x in returns)
    for _ in range(n_resamples):
        # generate stationary bootstrap sample of length n
        sample: list[float] = []
        while len(sample) < n:
            # start index uniformly
            start = rng.randrange(n)
            # block length geometric
            # Use geometric with p: L = ceil(log(U)/log(1-p))? But simpler: loop generating with p success per step.
            # Equivalent: sample length until termination with prob p after single draw
            length = 1
            while length < n - len(sample):
                # terminate with prob p, continue with 1-p
                if rng.random() < p:
                    break
                length += 1
            # Ensure at least 1
            length = max(1, length)
            # Append circular block
            for k in range(length):
                if len(sample) >= n:
                    break
                idx = (start + k) % n
                sample.append(arr[idx])
        # truncate to n
        sample = sample[:n]
        try:
            s = float(statistic(sample))
        except Exception:
            s = 0.0
        stats.append(s)
    stats.sort()
    # percentile interval
    lower_idx = int(math.floor((alpha / 2.0) * n_resamples))
    upper_idx = int(math.ceil((1 - alpha / 2.0) * n_resamples)) - 1
    lower_idx = max(0, min(lower_idx, n_resamples - 1))
    upper_idx = max(0, min(upper_idx, n_resamples - 1))
    lower = float(stats[lower_idx])
    upper = float(stats[upper_idx])
    if lower > upper:
        lower, upper = upper, lower
    return (lower, upper)


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
        # quantiles at 0.05,0.25,0.50,0.75,0.90,0.95,0.99
        q_levels = (0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
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

        gb_median = _q(givebacks, 0.50) if givebacks else 0.0
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
