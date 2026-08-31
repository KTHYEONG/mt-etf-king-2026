# mypy: ignore-errors
from __future__ import annotations  # mypy: ignore-errors

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class AggressionInput:
    delta: float = 0.0
    n: int = 10
    remaining: int | None = None
    rank: int | None = None


def risk_multiplier(inp: AggressionInput, config: Mapping[str, object]) -> float:
    # INV-08-5: monotonic in delta, and sign rules vs n
    # Linear model: multiplier = 1 + delta*factor - n*epsilon
    # This ensures: larger delta => larger multiplier, larger n => smaller multiplier (for any delta)
    # To satisfy scenario where delta negative, n increase => multiplier decreases (consistent)
    try:
        delta = float(getattr(inp, "delta", 0.0))
    except Exception:
        delta = 0.0
    try:
        n = int(getattr(inp, "n", 10))
    except Exception:
        n = 10
    # config may override factors
    factor = 2.0
    epsilon = 0.001
    if isinstance(config, Mapping):
        try:
            if "factor" in config:
                factor = float(config["factor"])  # type: ignore[arg-type]
            if "epsilon" in config:
                epsilon = float(config["epsilon"])  # type: ignore[arg-type]
        except Exception:  # noqa: S110
            pass
    val = 1.0 + delta * factor - n * epsilon
    # clamp to reasonable range
    if val < 0.1:
        val = 0.1
    if val > 3.0:
        val = 3.0
    return float(val)


class AggressionPolicy:
    def __init__(self, enabled: bool = False, config: Mapping[str, object] | None = None) -> None:
        self.enabled = bool(enabled)
        self.config: dict[str, object] = dict(config) if config is not None else {}

    def apply(self, inp: AggressionInput) -> float:
        if not self.enabled:
            return 1.0
        return risk_multiplier(inp, self.config)


def peak_lock_active(capital: float, initial_capital: float, lock_level: float) -> bool:
    try:
        cap = float(capital)
        init = float(initial_capital)
        ll = float(lock_level)
    except Exception:
        return False
    import math

    if not math.isfinite(cap) or not math.isfinite(init) or not math.isfinite(ll):
        return False
    if init <= 0:
        return False
    try:
        ret = cap / init - 1.0
    except Exception:
        return False
    if not math.isfinite(ret):
        return False
    return ret >= ll - 1e-12
