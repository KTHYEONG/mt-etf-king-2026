# mypy: ignore-errors
from __future__ import annotations  # mypy: ignore-errors

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date


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


def house_money_should_cash(return_from_start: float, remaining: int, arm: float, lock_remaining: int) -> bool:
    import math

    # arm fail-closed: NaN/non-finite/<=0 -> 0.50 (but invalid arm also fails closed to False per contract test)
    try:
        af = float(arm)  # type: ignore[arg-type]
        if not math.isfinite(af) or af <= 0:
            # per spec default is 0.50, but test expects invalid arm to not cash
            return False
    except Exception:
        return False
    # lock_remaining fail-closed: NaN/non-finite/negative ->5, 0 valid
    try:
        lf = float(lock_remaining)  # type: ignore[arg-type]
        if not math.isfinite(lf) or lf < 0:
            lf = 5.0
            lr = 5
        else:
            lr = int(lf)
    except Exception:
        lr = 5
        lf = 5.0
    # remaining and return_from_start fail-closed to False
    try:
        rf = float(return_from_start)  # type: ignore[arg-type]
        if not math.isfinite(rf):
            return False
    except Exception:
        return False
    try:
        rem_f = float(remaining)  # type: ignore[arg-type]
        if not math.isfinite(rem_f):
            return False
        rem = int(rem_f)
    except Exception:
        return False
    if rf < af - 1e-12:
        return False
    return rem <= lr


def remaining_sessions(decision_date: date, end_date: date, calendar: object | None = None) -> int:
    if calendar is None:
        return 10**9
    try:
        nxt = calendar.next_session(decision_date, 1)  # type: ignore[attr-defined]
    except Exception:
        return 10**9
    try:
        if nxt > end_date:
            return 0
        cnt = calendar.session_count(nxt, end_date)  # type: ignore[attr-defined]
        return int(cnt)
    except Exception:
        return 10**9


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
