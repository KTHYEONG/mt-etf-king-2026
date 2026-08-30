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


@dataclass(frozen=True)
class TailConcentrationConfig:
    enabled: bool = False
    tradable_states: frozenset[str] = frozenset({"LEADING", "RECOVERY"})
    risk_on_regimes: frozenset[str] = frozenset({"RISK_ON", "STRONG_RISK_ON"})
    w_top_full: float = 1.0
    k_full: int = 1


def tail_concentration_weights(
    scores: Mapping[str, float],
    base_config: ConfidenceSizingConfig,
    tail_config: TailConcentrationConfig,
    theme_states: Mapping[str, str] | None,
    regime: str | None,
) -> dict[str, float]:
    if not scores:
        return {}
    if tail_config is None or not bool(getattr(tail_config, "enabled", False)):
        return confidence_weights(scores, base_config)
    try:
        risk_on = frozenset(str(x) for x in getattr(tail_config, "risk_on_regimes", frozenset({"RISK_ON", "STRONG_RISK_ON"})))
    except Exception:
        risk_on = frozenset({"RISK_ON", "STRONG_RISK_ON"})
    regime_str = str(regime) if regime is not None else None
    if regime_str not in risk_on:
        return confidence_weights(scores, base_config)
    if not isinstance(theme_states, Mapping) or not theme_states:
        return confidence_weights(scores, base_config)
    try:
        tradable = frozenset(str(x) for x in getattr(tail_config, "tradable_states", frozenset({"LEADING", "RECOVERY"})))
    except Exception:
        tradable = frozenset({"LEADING", "RECOVERY"})
    sorted_items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_ticker = sorted_items[0][0]
    top_state = theme_states.get(top_ticker)
    if top_state is None:
        return confidence_weights(scores, base_config)
    top_str = str(top_state)
    if top_str not in tradable:
        return confidence_weights(scores, base_config)
    if top_str not in {"LEADING", "RECOVERY"}:
        return confidence_weights(scores, base_config)
    try:
        conf = compute_confidence(scores)
        c0 = float(getattr(base_config, "c0", 0.027174))
    except Exception:
        return confidence_weights(scores, base_config)
    if conf < c0 - 1e-12:
        return confidence_weights(scores, base_config)
    try:
        w = float(getattr(tail_config, "w_top_full", 1.0))
    except Exception:
        w = 1.0
    return {top_ticker: float(w)}


def confidence_vehicle_gate(w_top: float, config: ConfidenceSizingConfig, vehicle_conf_min: float) -> bool:
    try:
        w_max = float(config.w_max)
    except Exception:
        w_max = 1.0
    try:
        thr = float(vehicle_conf_min) * w_max
    except Exception:
        thr = 0.85 * w_max
    return float(w_top) < thr


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


@dataclass(frozen=True)
class LotteryExposureConfig:
    enabled: bool = False
    risk_on_regimes: frozenset[str] = frozenset({"RISK_ON", "STRONG_RISK_ON"})
    w_top: float = 1.0
    max_gross: float = 2.0
    suppress_vehicle_gate: bool = True
    suppress_trim: bool = True

    @classmethod
    def from_yaml(cls, raw: Mapping[str, object]) -> LotteryExposureConfig:
        # fail-closed defaults on missing/malformed
        enabled = False
        risk_on_regimes: frozenset[str] = frozenset({"RISK_ON", "STRONG_RISK_ON"})
        w_top = 1.0
        max_gross = 2.0
        suppress_vehicle_gate = True
        suppress_trim = True
        if not isinstance(raw, Mapping):
            return cls()
        # enabled
        try:
            if "enabled" in raw:
                enabled = bool(raw["enabled"])
        except Exception:
            enabled = False
        # risk_on_regimes
        try:
            if "risk_on_regimes" in raw:
                val = raw["risk_on_regimes"]
                if isinstance(val, (list, tuple, set, frozenset)):
                    risk_on_regimes = frozenset(str(x) for x in val)
                elif val is not None:
                    risk_on_regimes = frozenset({str(val)})
        except Exception:
            risk_on_regimes = frozenset({"RISK_ON", "STRONG_RISK_ON"})
        # w_top
        try:
            if "w_top" in raw:
                w_top = float(raw["w_top"])  # type: ignore[arg-type]
        except Exception:
            w_top = 1.0
        # max_gross
        try:
            if "max_gross" in raw:
                max_gross = float(raw["max_gross"])  # type: ignore[arg-type]
        except Exception:
            max_gross = 2.0
        # suppress_vehicle_gate
        try:
            if "suppress_vehicle_gate" in raw:
                suppress_vehicle_gate = bool(raw["suppress_vehicle_gate"])
        except Exception:
            suppress_vehicle_gate = True
        # suppress_trim
        try:
            if "suppress_trim" in raw:
                suppress_trim = bool(raw["suppress_trim"])
        except Exception:
            suppress_trim = True
        return cls(
            enabled=enabled,
            risk_on_regimes=risk_on_regimes,
            w_top=w_top,
            max_gross=max_gross,
            suppress_vehicle_gate=suppress_vehicle_gate,
            suppress_trim=suppress_trim,
        )


def lottery_active(
    regime: str | None,
    leverage_allowed: bool | None,
    config: LotteryExposureConfig | None,
) -> bool:
    if config is None:
        return False
    try:
        if not bool(getattr(config, "enabled", False)):
            return False
    except Exception:
        return False
    if leverage_allowed is not True:
        return False
    if regime is None:
        return False
    try:
        regimes = getattr(config, "risk_on_regimes", frozenset({"RISK_ON", "STRONG_RISK_ON"}))
        regime_str = str(regime)
        # normalize to set of strings
        try:
            regime_set = frozenset(str(x) for x in regimes)  # type: ignore[union-attr]
        except Exception:
            regime_set = frozenset({"RISK_ON", "STRONG_RISK_ON"})
        return regime_str in regime_set
    except Exception:
        return False


def lottery_concentration_weights(
    scores: Mapping[str, float],
    config: LotteryExposureConfig,
) -> dict[str, float]:
    if not scores:
        return {}
    # single key = argmax score tie-break ticker asc
    sorted_items = sorted(scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
    top_ticker = str(sorted_items[0][0])
    try:
        w = float(getattr(config, "w_top", 1.0))
    except Exception:
        w = 1.0
    # clamp [0,1]
    if w < 0:
        w = 0.0
    if w > 1:
        w = 1.0
    return {top_ticker: float(w)}
