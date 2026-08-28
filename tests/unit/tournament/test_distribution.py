"""SCENARIO-06-07 SCENARIO-06-08 SCENARIO-07P-05"""
from __future__ import annotations

import random
import statistics
from collections.abc import Sequence

from src.tournament.distribution import (
    ReturnDistribution,
    effective_sample_size,
    exceedance_curve,
    right_tail_score,
    stationary_bootstrap_ci,
)


def _iid_bootstrap_ci(
    returns: Sequence[float],
    statistic,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = random.Random(seed)  # noqa: S311
    base = list(returns)
    n = len(base)
    stats: list[float] = []
    for _ in range(n_resamples):
        sample = [base[rng.randrange(n)] for _ in range(n)]
        stats.append(float(statistic(sample)))
    stats.sort()
    lower_idx = int((alpha / 2.0) * n_resamples)
    upper_idx = int((1 - alpha / 2.0) * n_resamples) - 1
    return float(stats[lower_idx]), float(stats[upper_idx])


def test_SCENARIO_06_07_distribution_summary() -> None:  # noqa: N802
    """SCENARIO-06-07"""
    assert effective_sample_size(2090, 36) == 58

    exc = exceedance_curve([0.05, 0.15, 0.25, 0.35, 0.45], [0.10, 0.30, 0.50])
    assert exc == {0.10: 0.8, 0.30: 0.4, 0.50: 0.0}

    returns = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    weights = {0.75: 0.2, 0.90: 0.3, 0.95: 0.3, 0.99: 0.2}
    rts = right_tail_score(returns, weights)
    sorted_r = sorted(returns)
    n = len(sorted_r)

    def q(level: float) -> float:
        pos = level * (n - 1)
        lo = int(pos // 1)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return sorted_r[lo] * (1 - frac) + sorted_r[hi] * frac

    expected = 0.2 * q(0.75) + 0.3 * q(0.90) + 0.3 * q(0.95) + 0.2 * q(0.99)
    assert abs(rts - expected) < 1e-12

    dist = ReturnDistribution.summarise(
        name="B1",
        returns=returns,
        horizon=36,
        thresholds=[0.10, 0.30, 0.50],
        tail_weights=weights,
    )
    assert dist.n_effective == effective_sample_size(len(returns), 36)
    assert not hasattr(dist, "cagr")
    assert not hasattr(dist, "sharpe")


def test_SCENARIO_06_08_stationary_bootstrap_ci() -> None:  # noqa: N802
    """SCENARIO-06-08"""
    returns = [0.02] * 40 + [-0.02] * 40
    stat = statistics.mean

    ci_a = stationary_bootstrap_ci(returns, stat, expected_block=36, seed=7)
    ci_b = stationary_bootstrap_ci(returns, stat, expected_block=36, seed=7)
    assert ci_a == ci_b
    assert ci_a[0] < ci_a[1]

    iid_ci = _iid_bootstrap_ci(returns, stat, seed=7)
    stationary_width = ci_a[1] - ci_a[0]
    iid_width = iid_ci[1] - iid_ci[0]
    assert stationary_width > iid_width


def test_SCENARIO_07P_05_giveback_quantiles() -> None:  # noqa: N802
    """SCENARIO-07P-05"""
    dist = ReturnDistribution.summarise(name="x", returns=[0.0] * 5, horizon=5, thresholds=[0.1], tail_weights={}, givebacks=[0.0, 0.10, 0.20, 0.30, 0.40])
    assert abs(dist.giveback_median - 0.20) < 1e-12
    assert abs(dist.giveback_q90 - 0.36) < 1e-12
    dist2 = ReturnDistribution.summarise(name="x", returns=[0.0] * 5, horizon=5, thresholds=[0.1], tail_weights={})
    assert dist2.giveback_median == 0.0
    assert dist2.giveback_q90 == 0.0
    assert not hasattr(dist, "sharpe")
