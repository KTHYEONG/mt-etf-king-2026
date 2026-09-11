"""Phase 1 research-only P27 activation state on expanding PIT terciles of KOSPI mom60."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

import polars as pl

from src.features.pit import PitViolationError


class ActivationState(StrEnum):
    AGGRESSIVE_ON = "AGGRESSIVE_ON"
    UNCERTAIN = "UNCERTAIN"
    AGGRESSIVE_OFF = "AGGRESSIVE_OFF"


ACTIVATION_STATE_IS_PRODUCTION_GATE: Final[bool] = False
ACTIVATION_TERCILE_MIN_HISTORY: Final[int] = 252


@dataclass(frozen=True, slots=True)
class ActivationSnapshot:
    as_of: date
    state: ActivationState
    mom60: float | None
    q1: float | None
    q2: float | None
    n_history: int


def _linear_quantile(ordered: list[float], q: float) -> float:
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def classify_activation_state(
    *,
    decision_date: date,
    mom60_by_date: Mapping[date, float | None],
    min_history: int = ACTIVATION_TERCILE_MIN_HISTORY,
) -> ActivationSnapshot:
    if min_history < 1:
        raise ValueError(f"min_history must be >= 1, got {min_history}")
    for key in mom60_by_date:
        if key > decision_date:
            raise PitViolationError(f"PIT violation: mom60 date {key} exceeds decision_date {decision_date}")
    pairs = sorted(
        (
            (day, float(value))
            for day, value in mom60_by_date.items()
            if day <= decision_date and value is not None and math.isfinite(float(value))
        ),
    )
    values = [value for _, value in pairs]
    n_history = len(values)
    current: float | None = None
    for day, value in pairs:
        if day == decision_date:
            current = value
    if current is None or n_history < min_history:
        return ActivationSnapshot(
            as_of=decision_date, state=ActivationState.UNCERTAIN, mom60=None, q1=None, q2=None, n_history=n_history
        )
    ordered = sorted(values)
    q1 = _linear_quantile(ordered, 1.0 / 3.0)
    q2 = _linear_quantile(ordered, 2.0 / 3.0)
    if current > q2:
        state = ActivationState.AGGRESSIVE_ON
    elif current < q1:
        state = ActivationState.AGGRESSIVE_OFF
    else:
        state = ActivationState.UNCERTAIN
    return ActivationSnapshot(as_of=decision_date, state=state, mom60=current, q1=q1, q2=q2, n_history=n_history)


def classify_activation_series(
    mom60_by_date: Mapping[date, float | None],
    *,
    min_history: int = ACTIVATION_TERCILE_MIN_HISTORY,
) -> tuple[ActivationSnapshot, ...]:
    ordered_days = sorted(mom60_by_date.keys())
    snapshots: list[ActivationSnapshot] = []
    for as_of in ordered_days:
        cut = {day: mom60_by_date[day] for day in ordered_days if day <= as_of}
        snapshots.append(classify_activation_state(decision_date=as_of, mom60_by_date=cut, min_history=min_history))
    return tuple(snapshots)


def _normalize_state(value: ActivationState | str) -> str:
    if isinstance(value, ActivationState):
        return value.value
    return str(value)


def activation_conditional_table(
    *,
    states: Sequence[ActivationState | str],
    terminal_returns: Sequence[float],
    oracle_returns: Sequence[float],
    ruin_threshold: float = -0.25,
    overlap_horizon: int = 36,
) -> pl.DataFrame:
    if not (len(states) == len(terminal_returns) == len(oracle_returns)):
        raise ValueError(
            f"length mismatch: states={len(states)} terminals={len(terminal_returns)} oracle={len(oracle_returns)}"
        )
    order = (ActivationState.AGGRESSIVE_ON, ActivationState.UNCERTAIN, ActivationState.AGGRESSIVE_OFF)
    normalized = [_normalize_state(item) for item in states]
    terminals = [float(value) for value in terminal_returns]
    oracles = [float(value) for value in oracle_returns]
    rows: list[dict[str, object]] = []
    for bucket in order:
        idx = [i for i, name in enumerate(normalized) if name == bucket.value]
        n_windows = len(idx)
        n_effective = float(n_windows) / float(overlap_horizon)
        if n_windows == 0:
            rows.append(
                {
                    "state": bucket.value,
                    "n_windows": 0,
                    "n_effective": 0.0,
                    "p27_p30": 0.0,
                    "p27_p40": 0.0,
                    "p27_p45": 0.0,
                    "p27_p50": 0.0,
                    "p27_ruin25": 0.0,
                    "executable_oracle_p50": 0.0,
                    "capture_50": 0.0,
                }
            )
            continue
        bucket_terminals = [terminals[i] for i in idx]
        bucket_oracles = [oracles[i] for i in idx]
        p30 = sum(1 for v in bucket_terminals if v > 0.30) / n_windows
        p40 = sum(1 for v in bucket_terminals if v > 0.40) / n_windows
        p45 = sum(1 for v in bucket_terminals if v > 0.45) / n_windows
        p50 = sum(1 for v in bucket_terminals if v > 0.50) / n_windows
        ruin = sum(1 for v in bucket_terminals if v < ruin_threshold) / n_windows
        oracle_hits = sum(1 for v in bucket_oracles if v > 0.50)
        oracle_p50 = oracle_hits / n_windows
        if oracle_hits == 0:
            capture = 0.0
        else:
            capture = (
                sum(1 for t, o in zip(bucket_terminals, bucket_oracles, strict=True) if t > 0.50 and o > 0.50)
                / oracle_hits
            )
        rows.append(
            {
                "state": bucket.value,
                "n_windows": n_windows,
                "n_effective": n_effective,
                "p27_p30": float(p30),
                "p27_p40": float(p40),
                "p27_p45": float(p45),
                "p27_p50": float(p50),
                "p27_ruin25": float(ruin),
                "executable_oracle_p50": float(oracle_p50),
                "capture_50": float(capture),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "state": pl.String,
            "n_windows": pl.Int64,
            "n_effective": pl.Float64,
            "p27_p30": pl.Float64,
            "p27_p40": pl.Float64,
            "p27_p45": pl.Float64,
            "p27_p50": pl.Float64,
            "p27_ruin25": pl.Float64,
            "executable_oracle_p50": pl.Float64,
            "capture_50": pl.Float64,
        },
        strict=False,
    )
