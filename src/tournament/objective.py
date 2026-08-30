# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.tournament.distribution import ReturnDistribution, ruin_probability


@dataclass(frozen=True)
class ObjectiveGateConfig:
    g1_prob_threshold: float
    g1_min_improvement: float
    g2a_ruin_threshold: float
    g2a_max_prob: float

    @classmethod
    def from_yaml(cls, path: Path) -> ObjectiveGateConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        gates = raw.get("gates") if isinstance(raw, dict) else None
        if gates is None:
            gates = raw if isinstance(raw, dict) else {}
        # Expected keys: g1_prob_threshold, g1_min_improvement, g2a_ruin_threshold, g2a_max_prob
        g1_prob = float(gates.get("g1_prob_threshold", 0.30))
        g1_min = float(gates.get("g1_min_improvement", 0.02))
        g2a_thresh = float(gates.get("g2a_ruin_threshold", -0.25))
        g2a_max = float(gates.get("g2a_max_prob", 0.05))
        return cls(g1_prob_threshold=g1_prob, g1_min_improvement=g1_min, g2a_ruin_threshold=g2a_thresh, g2a_max_prob=g2a_max)


@dataclass(frozen=True)
class ObjectiveGateResult:
    status: str
    failures: tuple[str, ...]
    p_tail: float
    baseline_p_tail: float
    ruin_probability: float
    paired_delta_ci: tuple[float, float] | None


def paired_tail_delta_ci(
    candidate_returns: Sequence[float],
    control_returns: Sequence[float],
    threshold: float,
    *,
    expected_block: int,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    if not candidate_returns or not control_returns:
        return (0.0, 0.0)
    n = min(len(candidate_returns), len(control_returns))
    if n == 0:
        return (0.0, 0.0)
    if expected_block <= 0:
        raise ValueError("expected_block must be >0")
    p = 1.0 / float(expected_block)
    rng = random.Random(seed)
    # Build paired differences of indicator
    diffs: list[float] = []
    for i in range(n):
        c = float(candidate_returns[i]) > float(threshold)
        b = float(control_returns[i]) > float(threshold)
        diffs.append(float(int(c) - int(b)))
    # stationary bootstrap on diffs circular
    base = float(sum(diffs) / len(diffs)) if diffs else 0.0
    stats: list[float] = []
    arr = list(diffs)
    for _ in range(n_resamples):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randrange(n)
            length = 1
            while length < n - len(sample):
                if rng.random() < p:
                    break
                length += 1
            length = max(1, length)
            for k in range(length):
                if len(sample) >= n:
                    break
                idx = (start + k) % n
                sample.append(arr[idx])
        sample = sample[:n]
        try:
            s = float(sum(sample) / len(sample)) if sample else 0.0
        except Exception:
            s = 0.0
        stats.append(s)
    stats.sort()
    lower_idx = int(math.floor(0.025 * n_resamples))
    upper_idx = int(math.ceil(0.975 * n_resamples)) - 1
    lower_idx = max(0, min(lower_idx, n_resamples - 1))
    upper_idx = max(0, min(upper_idx, n_resamples - 1))
    lower = float(stats[lower_idx])
    upper = float(stats[upper_idx])
    if lower > upper:
        lower, upper = upper, lower
    return (lower, upper)


def evaluate_objective_gates(
    candidate: ReturnDistribution,
    baseline_b0: ReturnDistribution,
    config: ObjectiveGateConfig,
    *,
    paired_delta_ci: tuple[float, float] | None = None,
) -> ObjectiveGateResult:
    # INSUFFICIENT handling
    if candidate.n_effective < 1 or baseline_b0.n_effective < 1:
        return ObjectiveGateResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=("INSUFFICIENT_SAMPLE",),
            p_tail=0.0,
            baseline_p_tail=0.0,
            ruin_probability=1.0,
            paired_delta_ci=paired_delta_ci,
        )
    # Check missing lineage/windows artifacts? For this function, we consider paired_delta_ci None but candidate returns maybe empty?
    # Actually if candidate returns empty, treat insufficient
    if not candidate.returns or not baseline_b0.returns:
        return ObjectiveGateResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=("MISSING_ARTIFACT",),
            p_tail=0.0,
            baseline_p_tail=0.0,
            ruin_probability=1.0,
            paired_delta_ci=paired_delta_ci,
        )
    # Compute P(R > g1 threshold)
    thresh = float(config.g1_prob_threshold)
    # exceedance probability
    p_tail = float(sum(1 for r in candidate.returns if float(r) > thresh) / len(candidate.returns)) if candidate.returns else 0.0
    baseline_p = float(sum(1 for r in baseline_b0.returns if float(r) > thresh) / len(baseline_b0.returns)) if baseline_b0.returns else 0.0
    ruin_prob = ruin_probability(candidate.returns, float(config.g2a_ruin_threshold))
    failures: list[str] = []
    # G1: candidate P >= baseline + min_improvement
    if not (p_tail >= baseline_p + float(config.g1_min_improvement) - 1e-12):
        failures.append("G1_TAIL")
    # G2a: ruin <= max_prob
    # If ruin > max, fail with G2A_RUIN
    if not (ruin_prob <= float(config.g2a_max_prob) + 1e-12):
        failures.append("G2A_RUIN")
    status = "PASS" if not failures else "FAIL"
    # If failures empty but missing artifacts would have been handled above, return PASS else FAIL
    return ObjectiveGateResult(
        status=status,
        failures=tuple(failures),
        p_tail=float(p_tail),
        baseline_p_tail=float(baseline_p),
        ruin_probability=float(ruin_prob),
        paired_delta_ci=paired_delta_ci,
    )


@dataclass(frozen=True)
class P15AdoptionReport:
    status: str
    failures: tuple[str, ...]
    objective: ObjectiveGateResult | None = None


def _dist_exceedance(dist: ReturnDistribution, threshold: float) -> float:
    if not dist.exceedance:
        return 0.0
    direct = dist.exceedance.get(threshold)
    if direct is not None:
        return float(direct)
    for key, value in dist.exceedance.items():
        try:
            if abs(float(key) - float(threshold)) < 1e-9:
                return float(value)
        except Exception:
            continue
    return 0.0


def evaluate_p15_adoption_report(
    *,
    p15: ReturnDistribution,
    b1: ReturnDistribution,
    b0: ReturnDistribution,
    p14: ReturnDistribution,
    config: ObjectiveGateConfig,
    max_positive_family_count: int,
    multi_family_rate: float,
    artifacts_complete: bool,
    leverage_scenarios: Sequence[str],
    era_p40_deltas: Mapping[str, float] | None = None,
    era_b1_tail_event_counts: Mapping[str, int] | None = None,
) -> P15AdoptionReport:
    failures: list[str] = []
    if not artifacts_complete:
        failures.append("MISSING_ARTIFACT")
    if (
        p15.n_effective < 1
        or b1.n_effective < 1
        or b0.n_effective < 1
        or p14.n_effective < 1
        or not p15.returns
        or not b1.returns
        or not b0.returns
        or not p14.returns
    ):
        return P15AdoptionReport(
            status="INSUFFICIENT_EVIDENCE",
            failures=tuple(failures or ("INSUFFICIENT_SAMPLE",)),
            objective=None,
        )
    if set(leverage_scenarios) < {"aggressive", "conservative"}:
        failures.append("LEVERAGE_SCENARIOS")
    p30_p15 = _dist_exceedance(p15, 0.30)
    p40_p15 = _dist_exceedance(p15, 0.40)
    p30_b1 = _dist_exceedance(b1, 0.30)
    p40_b1 = _dist_exceedance(b1, 0.40)
    if p40_p15 < p40_b1 + 0.02 - 1e-12:
        failures.append("P40_VS_B1")
    if p30_p15 < p30_b1 - 1e-12:
        failures.append("P30_VS_B1")
    ruin_p15 = ruin_probability(p15.returns, config.g2a_ruin_threshold)
    ruin_p14 = ruin_probability(p14.returns, config.g2a_ruin_threshold)
    if ruin_p15 > float(config.g2a_max_prob) + 1e-12:
        failures.append("G2A_RUIN")
    if ruin_p15 > ruin_p14 + 0.01 + 1e-12:
        failures.append("RUIN_VS_P14")
    if max_positive_family_count > 1:
        failures.append("FAMILY_COUNT")
    if multi_family_rate > 1e-12:
        failures.append("MULTI_FAMILY")
    if era_p40_deltas and era_b1_tail_event_counts:
        for era, delta in era_p40_deltas.items():
            if int(era_b1_tail_event_counts.get(era, 0)) >= 5 and float(delta) < -1e-12:
                failures.append(f"ERA_{era}")
    objective = evaluate_objective_gates(p15, b0, config)
    if objective.status == "INSUFFICIENT_EVIDENCE":
        failures.append("INSUFFICIENT_SAMPLE")
    elif objective.status == "FAIL":
        failures.extend(list(objective.failures))
    if failures:
        status = (
            "INSUFFICIENT_EVIDENCE"
            if all(f in {"MISSING_ARTIFACT", "INSUFFICIENT_SAMPLE"} for f in failures)
            else "FAIL"
        )
        return P15AdoptionReport(status=status, failures=tuple(failures), objective=objective)
    return P15AdoptionReport(status="PASS", failures=(), objective=objective)


@dataclass(frozen=True)
class P16AdoptionReport:
    status: str
    failures: tuple[str, ...]
    objective: ObjectiveGateResult | None = None


def evaluate_p16_adoption_report(
    *,
    p16: ReturnDistribution,
    b1: ReturnDistribution,
    b0: ReturnDistribution,
    p14: ReturnDistribution,
    config: ObjectiveGateConfig,
    artifacts_complete: bool,
    leverage_scenarios: Sequence[str],
    skip_capacity_violations: int,
    vehicle_mult2_rate: float,
) -> P16AdoptionReport:
    failures: list[str] = []
    if not artifacts_complete:
        failures.append("MISSING_ARTIFACT")
    if (
        p16.n_effective < 1
        or b1.n_effective < 1
        or b0.n_effective < 1
        or p14.n_effective < 1
        or not p16.returns
        or not b1.returns
        or not b0.returns
        or not p14.returns
    ):
        return P16AdoptionReport(
            status="INSUFFICIENT_EVIDENCE",
            failures=tuple(failures or ("INSUFFICIENT_SAMPLE",)),
            objective=None,
        )
    if not {"aggressive", "conservative"} <= set(leverage_scenarios):
        failures.append("LEVERAGE_SCENARIOS")
    p30_p16 = _dist_exceedance(p16, 0.30)
    p40_p16 = _dist_exceedance(p16, 0.40)
    p50_p16 = _dist_exceedance(p16, 0.50)
    p30_b1 = _dist_exceedance(b1, 0.30)
    p40_b1 = _dist_exceedance(b1, 0.40)
    p50_b1 = _dist_exceedance(b1, 0.50)
    if p30_p16 < p30_b1 - 1e-12:
        failures.append("P30_VS_B1")
    if p40_p16 < p40_b1 - 1e-12:
        failures.append("P40_VS_B1")
    if p50_p16 < p50_b1 - 1e-12:
        failures.append("P50_VS_B1")
    ruin_p16 = ruin_probability(p16.returns, config.g2a_ruin_threshold)
    ruin_p14 = ruin_probability(p14.returns, config.g2a_ruin_threshold)
    if ruin_p16 > float(config.g2a_max_prob) + 1e-12:
        failures.append("G2A_RUIN")
    if ruin_p16 > ruin_p14 + 0.01 + 1e-12:
        failures.append("RUIN_VS_P14")
    if int(skip_capacity_violations) > 0:
        failures.append("CAPACITY_ON_CONVEXITY")
    if float(vehicle_mult2_rate) < 0.25 - 1e-12:
        failures.append("VEHICLE_ACTIVITY")
    objective = evaluate_objective_gates(p16, b0, config)
    if objective.status == "INSUFFICIENT_EVIDENCE":
        failures.append("INSUFFICIENT_SAMPLE")
    elif objective.status == "FAIL":
        failures.extend(list(objective.failures))
    if failures:
        status = (
            "INSUFFICIENT_EVIDENCE"
            if all(f in {"MISSING_ARTIFACT", "INSUFFICIENT_SAMPLE"} for f in failures)
            else "FAIL"
        )
        return P16AdoptionReport(status=status, failures=tuple(failures), objective=objective)
    return P16AdoptionReport(status="PASS", failures=(), objective=objective)
