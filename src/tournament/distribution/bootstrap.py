# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

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
