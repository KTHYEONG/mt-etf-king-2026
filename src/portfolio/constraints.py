from __future__ import annotations

from collections.abc import Mapping


class WeightViolationError(ValueError):
    pass


def normalize_weights(
    weights: Mapping[str, float],
    max_weight: float = 1.0,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    # Check negative and max_weight
    for ticker, w in weights.items():
        wf = float(w)
        if wf < -1e-12:
            raise WeightViolationError(f"negative weight {ticker}={wf}")
        if wf > float(max_weight) + 1e-12:
            raise WeightViolationError(f"weight {ticker}={wf} exceeds max_weight {max_weight}")
    total = sum(float(v) for v in weights.values())
    if total < -1e-12:
        raise WeightViolationError(f"negative total {total}")
    if total > 1.0 + tolerance:
        raise WeightViolationError(f"sum {total} exceeds 1.0 + tolerance {tolerance}")
    # The sum of instrument weights plus cash equals 1.0 within tolerance.
    # Cash is 1 - total
    cash = 1.0 - total
    if cash < -tolerance:
        raise WeightViolationError(f"cash {cash} negative beyond tolerance")
    # If total is too far from 1 and cash would be > tolerance? Actually requirement: sum + cash ==1 exactly.
    # The above ensures sum in [0, 1+tolerance] and cash >= -tolerance => sum <=1+tolerance and sum >= -tolerance?
    # Need also ensure sum not too low? For e.g. sum=0.6 cash=0.4 sum+cash=1 correct. So any sum <=1+tolerance is ok.
    # But test expects normalize_weights({'A':0.6,'B':0.3}) sum=0.9 cash 0.1 passes.
    # And {'A':0.7,'B':0.5} sum=1.2 >1 => raise.
    # So we just need to enforce sum <=1+tolerance.
    # Also if max_weight violated already raised.
    return {k: float(v) for k, v in weights.items()}
