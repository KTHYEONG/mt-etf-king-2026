from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class PortfolioIntent:
    kind: Literal["hold", "target", "cash"]
    weights: dict[str, float] = field(default_factory=dict)


HOLD_INTENT: PortfolioIntent = PortfolioIntent(kind="hold", weights={})
CASH_INTENT: PortfolioIntent = PortfolioIntent(kind="cash", weights={})


def resolve_portfolio_intent(
    alloc_result: Mapping[str, float] | PortfolioIntent | object | None,
    *,
    current_weights: Mapping[str, float],
    score_failed: bool,
) -> PortfolioIntent:
    if bool(score_failed):
        return HOLD_INTENT
    if alloc_result is None:
        return HOLD_INTENT
    if isinstance(alloc_result, PortfolioIntent):
        return alloc_result
    # handle objects with .weights attribute (e.g., PortfolioPolicy result)
    if hasattr(alloc_result, "weights"):  # noqa: B009
        try:
            w_raw = getattr(alloc_result, "weights")  # noqa: B009
            if isinstance(w_raw, Mapping):
                if len(w_raw) == 0:
                    return HOLD_INTENT
                w = {str(k): float(v) for k, v in dict(w_raw).items()}
                return PortfolioIntent(kind="target", weights=w)
        except Exception:  # noqa: S110
            pass
    if isinstance(alloc_result, Mapping):
        if len(alloc_result) == 0:
            return HOLD_INTENT
        try:
            w = {str(k): float(v) for k, v in dict(alloc_result).items()}
        except Exception:
            return HOLD_INTENT
        return PortfolioIntent(kind="target", weights=w)
    return HOLD_INTENT
