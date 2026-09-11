# mypy: ignore-errors
# ruff: noqa
"""Championship cutoff-AUC objective and attack policy (live production gate)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from src.core.config import config_value

CUTOFF_AUC_THRESHOLDS: Final[tuple[float, ...]] = tuple(
    config_value("gates", "cutoff_auc", "thresholds", required=True)
)
CUTOFF_AUC_LOW: Final[float] = CUTOFF_AUC_THRESHOLDS[0]
CUTOFF_AUC_HIGH: Final[float] = CUTOFF_AUC_THRESHOLDS[3]
CUTOFF_AUC_IS_PRODUCTION_GATE: Final[bool] = True
INACTIVE_PARTICIPATION_IS_PRODUCTION_GATE: Final[bool] = False
ATTACK_RUIN_MAX: Final[float] = 0.05


@dataclass(frozen=True)
class AttackPolicyResult:
    status: str
    failures: tuple[str, ...]
    candidate_smooth_active: float
    incumbent_smooth_active: float
    capture: float
    inactive_false_positive_loss: float
    ruin_probability: float


def smooth_cutoff_utility(value: float, *, low: float = CUTOFF_AUC_LOW, high: float = CUTOFF_AUC_HIGH) -> float:
    fv = float(value)
    fl = float(low)
    fh = float(high)
    if not math.isfinite(fv) or not math.isfinite(fl) or not math.isfinite(fh):
        raise ValueError("non-finite value/low/high")
    if fh <= fl:
        raise ValueError("high must exceed low")
    if fv <= fl:
        return 0.0
    if fv >= fh:
        return 1.0
    return float((fv - fl) / (fh - fl))


def cutoff_auc_score(returns: Sequence[float], thresholds: Sequence[float] = CUTOFF_AUC_THRESHOLDS) -> float:
    ret = list(returns)
    thr = list(thresholds)
    if len(ret) == 0 or len(thr) == 0:
        raise ValueError("returns/thresholds empty")
    for v in ret:
        if not isinstance(v, (int, float)):
            raise ValueError("returns contains non-numeric")
        if not math.isfinite(float(v)):
            raise ValueError("returns contains non-finite")
    for t in thr:
        if not isinstance(t, (int, float)):
            raise ValueError("thresholds contains non-numeric")
        if not math.isfinite(float(t)):
            raise ValueError("thresholds contains non-finite")
    n = len(ret)
    total = 0.0
    for t in thr:
        ft = float(t)
        cnt = sum(1 for r in ret if float(r) > ft)
        total += float(cnt) / float(n)
    return float(total / float(len(thr)))


def mean_smooth_cutoff_utility(
    returns: Sequence[float], *, low: float = CUTOFF_AUC_LOW, high: float = CUTOFF_AUC_HIGH
) -> float:
    ret = list(returns)
    if len(ret) == 0:
        raise ValueError("returns empty")
    for v in ret:
        if not isinstance(v, (int, float)):
            raise ValueError("returns contains non-numeric")
        if not math.isfinite(float(v)):
            raise ValueError("returns contains non-finite")
    total = 0.0
    for v in ret:
        total += smooth_cutoff_utility(float(v), low=low, high=high)
    return float(total / float(len(ret)))


def opportunity_conditioned_capture(
    strategy: Sequence[float], oracle: Sequence[float], *, threshold: float = CUTOFF_AUC_THRESHOLDS[2]
) -> float:
    from src.research.feasibility_metrics import capture_intersection

    s = list(strategy)
    o = list(oracle)
    if len(s) != len(o):
        raise ValueError("length mismatch")
    return float(capture_intersection(s, o, float(threshold)).rate)


def inactive_false_positive_loss(returns: Sequence[float], active: Sequence[bool]) -> float:
    ret = list(returns)
    act = list(active)
    if len(ret) != len(act):
        raise ValueError("length mismatch")
    inactive = [float(r) for r, a in zip(ret, act) if not a]
    if not inactive:
        return 0.0
    total = 0.0
    for r in inactive:
        neg = -float(r)
        if neg > 0.0:
            total += neg
    return float(total / float(len(inactive)))


def inactive_activity_rate(returns: Sequence[float], active: Sequence[bool]) -> float:
    ret = list(returns)
    act = list(active)
    if len(ret) != len(act):
        raise ValueError("length mismatch")
    inactive = [float(r) for r, a in zip(ret, act) if not a]
    if not inactive:
        return 0.0
    cnt = sum(1 for r in inactive if abs(float(r)) > 1e-12)
    return float(cnt / float(len(inactive)))


def resolve_attack_sleeve(sleeve: str | None, *, inactive_participation: bool = False) -> str:
    if sleeve == "LOTTERY_ON":
        return "MOM60"
    if sleeve == "CRASH_REBOUND":
        return "REBOUND"
    if sleeve == "INACTIVE" and bool(inactive_participation):
        return "PARTICIPATE"
    return "CASH"


def apply_attack_sleeve_route(
    *,
    sleeve: str | None,
    mom60_scores: Mapping[str, float] | object,
    rebound_scores: Mapping[str, float] | object,
    production_gate: bool,
    inactive_scores: Mapping[str, float] | object | None = None,
    inactive_participation: bool = False,
) -> Mapping[str, float] | object:
    from src.portfolio.intent import CASH_INTENT

    if not production_gate:
        return mom60_scores
    route = resolve_attack_sleeve(sleeve, inactive_participation=inactive_participation)
    if route == "MOM60":
        return mom60_scores
    if route == "REBOUND":
        if isinstance(rebound_scores, Mapping) and len(rebound_scores) > 0:
            return rebound_scores
        return CASH_INTENT
    if route == "PARTICIPATE":
        if isinstance(inactive_scores, Mapping) and len(inactive_scores) > 0:
            return inactive_scores
        return CASH_INTENT
    return CASH_INTENT


def evaluate_attack_policy(
    *,
    candidate_returns: Sequence[float],
    incumbent_returns: Sequence[float],
    oracle_returns: Sequence[float],
    active: Sequence[bool],
    execution_parity: bool,
    gross_violation_count: int | None,
) -> AttackPolicyResult:
    from src.tournament.distribution import ruin_probability as _ruin_probability
    from src.tournament.objective.reports import GROSS_METRIC_UNAVAILABLE

    cand = list(candidate_returns)
    inc = list(incumbent_returns)
    orac = list(oracle_returns)
    act = list(active)
    if len(cand) == 0 or len(inc) == 0 or len(orac) == 0 or len(act) == 0:
        return AttackPolicyResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=("MISSING_ARTIFACT",),
            candidate_smooth_active=0.0,
            incumbent_smooth_active=0.0,
            capture=0.0,
            inactive_false_positive_loss=0.0,
            ruin_probability=0.0,
        )
    if not (len(cand) == len(inc) == len(orac) == len(act)):
        return AttackPolicyResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=("MISSING_ARTIFACT",),
            candidate_smooth_active=0.0,
            incumbent_smooth_active=0.0,
            capture=0.0,
            inactive_false_positive_loss=0.0,
            ruin_probability=0.0,
        )
    has_nonfinite = False
    for v in cand + inc + orac:
        if not isinstance(v, (int, float)):
            has_nonfinite = True
        elif not math.isfinite(float(v)):
            has_nonfinite = True
    if has_nonfinite:
        return AttackPolicyResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=("MISSING_ARTIFACT",),
            candidate_smooth_active=0.0,
            incumbent_smooth_active=0.0,
            capture=0.0,
            inactive_false_positive_loss=0.0,
            ruin_probability=0.0,
        )
    if gross_violation_count is None:
        return AttackPolicyResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=(GROSS_METRIC_UNAVAILABLE,),
            candidate_smooth_active=0.0,
            incumbent_smooth_active=0.0,
            capture=0.0,
            inactive_false_positive_loss=0.0,
            ruin_probability=0.0,
        )
    if not any(act):
        return AttackPolicyResult(
            status="INSUFFICIENT_EVIDENCE",
            failures=("INSUFFICIENT_ACTIVE",),
            candidate_smooth_active=0.0,
            incumbent_smooth_active=0.0,
            capture=0.0,
            inactive_false_positive_loss=0.0,
            ruin_probability=0.0,
        )
    failures: list[str] = []
    if not execution_parity:
        failures.append("EXECUTION_PARITY")
    if int(gross_violation_count) != 0:
        failures.append("GROSS_EXPOSURE")
    ruin = float(_ruin_probability(cand, -0.25))
    if ruin > float(ATTACK_RUIN_MAX):
        failures.append("RUIN")
    cand_active = [float(c) for c, a in zip(cand, act) if a]
    inc_active = [float(v) for v, a in zip(inc, act) if a]
    cand_smooth = float(sum(smooth_cutoff_utility(v) for v in cand_active) / float(len(cand_active)))
    inc_smooth = float(sum(smooth_cutoff_utility(v) for v in inc_active) / float(len(inc_active)))
    if cand_smooth < inc_smooth - 1e-12:
        failures.append("ACTIVE_UTILITY")
    if inactive_activity_rate(cand, act) > 0.0:
        failures.append("INACTIVE_ACTIVITY")
    if inactive_false_positive_loss(cand, act) > 0.0:
        failures.append("INACTIVE_LOSS")
    capture = float(opportunity_conditioned_capture(cand, orac, threshold=CUTOFF_AUC_THRESHOLDS[2]))
    loss = float(inactive_false_positive_loss(cand, act))
    if failures:
        return AttackPolicyResult(
            status="FAIL",
            failures=tuple(failures),
            candidate_smooth_active=float(cand_smooth),
            incumbent_smooth_active=float(inc_smooth),
            capture=float(capture),
            inactive_false_positive_loss=float(loss),
            ruin_probability=float(ruin),
        )
    return AttackPolicyResult(
        status="PASS",
        failures=(),
        candidate_smooth_active=float(cand_smooth),
        incumbent_smooth_active=float(inc_smooth),
        capture=float(capture),
        inactive_false_positive_loss=float(loss),
        ruin_probability=float(ruin),
    )
