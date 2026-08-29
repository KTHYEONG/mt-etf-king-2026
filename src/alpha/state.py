from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class ThemeMetrics:
    theme: str
    representative: str
    rs: float
    accel: float
    breadth: float
    ext: float
    dd: float


class ThemeState(StrEnum):
    DISCOVERY = "DISCOVERY"
    EMERGING = "EMERGING"
    LEADING = "LEADING"
    OVERHEATED = "OVERHEATED"
    BREAKDOWN = "BREAKDOWN"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True)
class TransitionConfig:
    rs_in: float
    rs_out: float
    rs_hi: float
    accel_in: float
    accel_out: float
    breadth_in: float
    breadth_out: float
    ext_in: float
    ext_out: float
    dd_in: float
    dd_out: float
    patience: int

    def validate(self) -> None:
        pairs = [
            ("rs_in", self.rs_in, "rs_out", self.rs_out),
            ("accel_in", self.accel_in, "accel_out", self.accel_out),
            ("breadth_in", self.breadth_in, "breadth_out", self.breadth_out),
            ("ext_in", self.ext_in, "ext_out", self.ext_out),
            ("dd_in", self.dd_in, "dd_out", self.dd_out),
        ]
        for a_name, a_val, b_name, b_val in pairs:
            if not (a_val > b_val):
                raise ValueError(f"hysteresis violation: {a_name} ({a_val}) must be > {b_name} ({b_val})")

    @classmethod
    def from_yaml(cls, raw: Mapping[str, object]) -> TransitionConfig:
        def _get(key: str, default: float | int | None = None) -> float | int:
            if key in raw:
                val = raw[key]
                return val  # type: ignore[return-value]
            if default is not None:
                return default
            raise KeyError(key)

        rs_in = float(_get("rs_in"))
        rs_out = float(_get("rs_out"))
        rs_hi = float(_get("rs_hi"))
        accel_in = float(_get("accel_in"))
        accel_out = float(_get("accel_out"))
        breadth_in = float(_get("breadth_in"))
        breadth_out = float(_get("breadth_out"))
        ext_in = float(_get("ext_in"))
        ext_out = float(_get("ext_out"))
        dd_in = float(_get("dd_in"))
        dd_out = float(_get("dd_out"))
        patience = int(_get("patience", 3))
        cfg = cls(
            rs_in=rs_in,
            rs_out=rs_out,
            rs_hi=rs_hi,
            accel_in=accel_in,
            accel_out=accel_out,
            breadth_in=breadth_in,
            breadth_out=breadth_out,
            ext_in=ext_in,
            ext_out=ext_out,
            dd_in=dd_in,
            dd_out=dd_out,
            patience=patience,
        )
        cfg.validate()
        return cfg


def transition(
    state: ThemeState,
    metrics: ThemeMetrics,
    config: TransitionConfig,
    patience_counter: int = 0,
    *,
    validate: bool = True,
) -> tuple[ThemeState, int]:
    rs = float(metrics.rs)
    accel = float(metrics.accel)
    breadth = float(metrics.breadth)
    ext = float(metrics.ext)
    dd = float(metrics.dd)

    if validate:
        config.validate()

    next_state = state
    next_counter = patience_counter

    if state == ThemeState.DISCOVERY:
        if rs > config.rs_in and accel > config.accel_in and breadth > config.breadth_in:
            next_state = ThemeState.EMERGING
            next_counter = 0
        else:
            next_state = ThemeState.DISCOVERY
            next_counter = 0
    elif state == ThemeState.EMERGING:
        if rs > config.rs_hi and breadth > config.breadth_in and accel > config.accel_in:
            next_state = ThemeState.LEADING
            next_counter = 0
        elif rs < config.rs_out or breadth < config.breadth_out or accel < config.accel_out:
            next_state = ThemeState.DISCOVERY
            next_counter = 0
        else:
            next_state = ThemeState.EMERGING
            next_counter = 0
    elif state == ThemeState.LEADING:
        if ext > config.ext_in:
            next_state = ThemeState.OVERHEATED
            next_counter = 0
        elif dd > config.dd_in:  # noqa: SIM114
            if patience_counter + 1 >= config.patience:
                next_state = ThemeState.BREAKDOWN
                next_counter = 0
            else:
                next_state = ThemeState.LEADING
                next_counter = patience_counter + 1
        elif rs < config.rs_out or breadth < config.breadth_out or accel < config.accel_out:  # noqa: SIM114
            if patience_counter + 1 >= config.patience:
                next_state = ThemeState.BREAKDOWN
                next_counter = 0
            else:
                next_state = ThemeState.LEADING
                next_counter = patience_counter + 1
        else:
            next_state = ThemeState.LEADING
            next_counter = 0
    elif state == ThemeState.OVERHEATED:
        if dd > config.dd_in:
            next_state = ThemeState.BREAKDOWN
            next_counter = 0
        elif ext < config.ext_out and rs > config.rs_out:
            next_state = ThemeState.LEADING
            next_counter = 0
        else:
            next_state = ThemeState.OVERHEATED
            next_counter = 0
    elif state == ThemeState.BREAKDOWN:
        if rs > config.rs_in and breadth > config.breadth_in and accel > config.accel_in:  # noqa: SIM114
            next_state = ThemeState.RECOVERY
            next_counter = 0
        elif rs > config.rs_out and breadth > config.breadth_out:  # noqa: SIM114
            next_state = ThemeState.RECOVERY
            next_counter = 0
        else:
            next_state = ThemeState.BREAKDOWN
            next_counter = 0
    elif state == ThemeState.RECOVERY:
        if rs > config.rs_in and breadth > config.breadth_in and accel > config.accel_in:
            next_state = ThemeState.LEADING
            next_counter = 0
        elif rs < config.rs_out or breadth < config.breadth_out:
            next_state = ThemeState.BREAKDOWN
            next_counter = 0
        else:
            next_state = ThemeState.RECOVERY
            next_counter = 0
    else:
        next_state = state
        next_counter = 0

    if state == ThemeState.BREAKDOWN and next_state == ThemeState.LEADING:
        next_state = ThemeState.RECOVERY

    return next_state, next_counter


def run_state_machine(
    metrics_series: Sequence[ThemeMetrics],
    config: TransitionConfig,
    initial: ThemeState = ThemeState.DISCOVERY,
) -> list[ThemeState]:
    states: list[ThemeState] = []
    cur = initial
    counter = 0
    for m in metrics_series:
        nxt, counter = transition(cur, m, config, patience_counter=counter)
        states.append(nxt)
        cur = nxt
    return states
