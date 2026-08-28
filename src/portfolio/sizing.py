from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum


class SizingScheme(StrEnum):
    TOP1 = "TOP1"
    TOP2_70_30 = "TOP2_70_30"
    TOP3_50_30_20 = "TOP3_50_30_20"
    EQUAL_K = "EQUAL_K"


def weights_from_scores(
    scores: Mapping[str, float],
    scheme: SizingScheme,
    k: int = 3,
) -> dict[str, float]:
    if not scores:
        return {}
    # deterministic sort: descending score, then ticker ascending
    sorted_items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if scheme == SizingScheme.TOP1:
        top = sorted_items[:1]
        weights = [1.0]
    elif scheme == SizingScheme.TOP2_70_30:
        # Use first 2 with 0.7,0.3 ; if fewer than 2, keep specified weights for available, rest cash
        full = [0.7, 0.3]
        n = min(2, len(sorted_items))
        top = sorted_items[:n]
        weights = full[:n]
    elif scheme == SizingScheme.TOP3_50_30_20:
        full = [0.5, 0.3, 0.2]
        n = min(3, len(sorted_items))
        top = sorted_items[:n]
        weights = full[:n]
    elif scheme == SizingScheme.EQUAL_K:
        n = min(int(k), len(sorted_items))
        if n <= 0:
            return {}
        top = sorted_items[:n]
        w = 1.0 / int(k)
        weights = [w] * n
        # If fewer than k, shortfall is cash -> no rescaling
    else:
        raise ValueError(f"unknown scheme {scheme}")
    result: dict[str, float] = {}
    for (ticker, _), w in zip(top, weights, strict=False):
        result[ticker] = float(w)
    return result
