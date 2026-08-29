# mypy: ignore-errors
from __future__ import annotations  # mypy: ignore-errors

import math
from collections.abc import Mapping
from dataclasses import dataclass
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
    sorted_items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if scheme == SizingScheme.TOP1:
        top = sorted_items[:1]
        weights = [1.0]
    elif scheme == SizingScheme.TOP2_70_30:
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
    else:
        raise ValueError(f"unknown scheme {scheme}")
    result: dict[str, float] = {}
    for (ticker, _), w in zip(top, weights, strict=False):
        result[ticker] = float(w)
    return result


def compute_confidence(scores: Mapping[str, float]) -> float:
    if not scores or len(scores) < 2:
        return 0.0
    sorted_vals = sorted((float(v) for v in scores.values()), reverse=True)
    return float(sorted_vals[0] - sorted_vals[1])


@dataclass(frozen=True)
class ConfidenceSizingConfig:
    w_min: float = 0.3333333333333333
    w_max: float = 1.0
    c0: float = 0.027174
    tau: float = 0.016073
    k: int = 3


def confidence_weights(scores: Mapping[str, float], config: ConfidenceSizingConfig) -> dict[str, float]:
    if not scores:
        return {}
    sorted_items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    conf = compute_confidence(scores)
    # logistic mapping with k as steepness multiplier
    try:
        # use k as steepness to ensure low conf yields w_min region
        eff = float(config.k) * (conf - float(config.c0)) / float(config.tau) if float(config.tau) != 0 else 0.0
    except Exception:
        eff = 0.0
    # sigmoid
    if eff > 500:
        sig = 1.0
    elif eff < -500:
        sig = 0.0
    else:
        sig = 1.0 / (1.0 + math.exp(-eff))
    w_top = float(config.w_min) + (float(config.w_max) - float(config.w_min)) * sig
    # clamp
    if w_top > 1.0:
        w_top = 1.0
    if w_top < 0.0:
        w_top = 0.0
    k_val = int(config.k)
    if k_val <= 0:
        k_val = 1
    n = min(k_val, len(sorted_items))
    if n <= 0:
        return {}
    if n == 1:
        return {sorted_items[0][0]: float(w_top)}
    residual = 1.0 - float(w_top)
    if residual < 0:
        residual = 0.0
        w_top = 1.0
    result: dict[str, float] = {}
    result[sorted_items[0][0]] = float(w_top)
    if n > 1:
        each = residual / (n - 1) if (n - 1) > 0 else 0.0
        for ticker, _ in sorted_items[1:n]:
            result[ticker] = float(each)
    # ensure sum <=1.0
    total = sum(result.values())
    if total > 1.0 + 1e-9:
        # scale down proportionally
        factor = 1.0 / total if total != 0 else 0
        for k_ in list(result.keys()):
            result[k_] = float(result[k_] * factor)
    return result
