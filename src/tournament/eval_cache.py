from __future__ import annotations

from collections.abc import Callable, Sequence


def protocol_cell_key(cost: object, participation: float) -> str:
    try:
        comm = float(getattr(cost, "commission_bps", 0) or 0)
    except Exception:
        comm = 0.0
    try:
        slip = float(getattr(cost, "slippage_bps", 0) or 0)
    except Exception:
        slip = 0.0
    try:
        part = float(participation)
    except Exception:
        part = 0.0
    c = round(comm * 100)  # noqa: RUF046
    s = round(slip * 100)  # noqa: RUF046
    p = round(part * 1000)  # noqa: RUF046
    return f"{int(c):04d}_{int(s):04d}_{int(p):04d}"


def is_canonical_protocol_cell(cost: object, participation: float) -> bool:
    from src.tournament.harness import (
        DEFAULT_PROTOCOL_COMMISSION_BPS,
        DEFAULT_PROTOCOL_PARTICIPATION,
        DEFAULT_PROTOCOL_SLIPPAGE_BPS,
    )

    try:
        comm = float(getattr(cost, "commission_bps", 0) or 0)
    except Exception:
        comm = 0.0
    try:
        slip = float(getattr(cost, "slippage_bps", 0) or 0)
    except Exception:
        slip = 0.0
    try:
        part = float(participation)
    except Exception:
        part = 0.0
    eps = 1e-9
    return abs(comm - float(DEFAULT_PROTOCOL_COMMISSION_BPS)) < eps and abs(slip - float(DEFAULT_PROTOCOL_SLIPPAGE_BPS)) < eps and abs(part - float(DEFAULT_PROTOCOL_PARTICIPATION)) < eps


def plan_control_evaluations(protocol: str, cases: Sequence[tuple[object, float]]) -> list[bool]:
    if protocol != "grid":
        return [True for _ in cases]
    return [is_canonical_protocol_cell(cost, part) for cost, part in cases]


class ControlRollingCache:
    def __init__(self) -> None:
        self.hits: int = 0
        self.misses: int = 0
        self._store: dict[str, object] = {}

    def get_or_run(self, key: str, factory: Callable[[], object]) -> object:
        if key in self._store:
            self.hits += 1
            return self._store[key]
        result = factory()
        self._store[key] = result
        self.misses += 1
        return result
