# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Mapping


class CompetitorField:
    def __init__(self, scenarios: Mapping[str, float] | None = None) -> None:
        if scenarios is not None:
            self.scenarios: dict[str, float] = {k: float(v) for k, v in scenarios.items()}
        else:
            self.scenarios = {"aggressive": 0.72, "normal": 0.478, "weak": 0.30}

    def rank_interval(self, own_return: float, n_competitors: int = 1000) -> dict[str, tuple[int, int]]:
        return rank_interval(own_return, self.scenarios, n_competitors)

    def interval(self, own_return: float, n_competitors: int = 1000) -> dict[str, tuple[int, int]]:
        return self.rank_interval(own_return, n_competitors)


def rank_interval(
    own_return: float,
    scenarios: Mapping[str, float],
    n_competitors: int = 1000,
) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    try:
        own = float(own_return)
    except Exception:
        own = 0.0
    try:
        n = int(n_competitors)
    except Exception:
        n = 1000
    if n <= 0:
        n = 1000
    for name, val in scenarios.items():
        try:
            thresh = float(val)
        except Exception:
            thresh = 0.0
        # Determine interval based on own vs thresh
        # If own >= thresh: good rank (1 .. 10%)
        # elif own >= thresh*0.5: mid rank
        # else low rank
        if own >= thresh:
            lo, hi = 1, max(1, n // 10)
        elif own >= thresh * 0.5:
            lo, hi = n // 10, n // 2
        else:
            lo, hi = n // 2, n
        if lo > hi:
            lo, hi = hi, lo
        # clamp
        lo = max(1, min(lo, n))
        hi = max(1, min(hi, n))
        if lo > hi:
            lo, hi = hi, lo
        out[str(name)] = (int(lo), int(hi))
    # Ensure at least aggressive/normal/weak if not provided
    for k in ("aggressive", "normal", "weak"):
        if k not in out:
            # provide default interval
            out[k] = (n // 10, n // 2)
    # Do NOT include win_probability per INV-08-7
    return out
