"""Return-space feasibility metrics (Phase 0, date-joined, fail-closed)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from src.research.decision_population import DecisionPopulation


class AlignError(ValueError):
    """Raised when strategy returns cannot be date-joined onto the population."""


class CaptureError(ValueError):
    """Raised when capture inputs are not comparable."""


@dataclass(frozen=True, slots=True)
class CaptureReport:
    n_oracle: int
    n_strategy: int
    n_both: int
    n_strategy_without_oracle: int
    rate: float


@dataclass(frozen=True, slots=True)
class GapReport:
    mean_selection_timing: float
    mean_giveback: float
    mean_total: float


def align_returns_by_entry_date(
    *,
    starts: Sequence[date],
    returns: Sequence[float],
    population: DecisionPopulation,
    require_complete: bool = True,
) -> tuple[float, ...]:
    if len(starts) != len(returns):
        raise AlignError(f"starts ({len(starts)}) and returns ({len(returns)}) length mismatch")
    by_start = {day: float(value) for day, value in zip(starts, returns, strict=True)}
    aligned: list[float] = []
    for window in population.windows:
        if window.entry_date not in by_start and require_complete:
            raise AlignError(f"missing return for entry_date {window.entry_date}")
        aligned.append(by_start[window.entry_date])
    return tuple(aligned)


def capture_intersection(strategy: Sequence[float], oracle: Sequence[float], threshold: float) -> CaptureReport:
    if len(strategy) != len(oracle):
        raise CaptureError(f"strategy ({len(strategy)}) and oracle ({len(oracle)}) length mismatch")
    over_strategy = [float(value) > threshold for value in strategy]
    over_oracle = [float(value) > threshold for value in oracle]
    n_oracle = sum(1 for hit in over_oracle if hit)
    n_strategy = sum(1 for hit in over_strategy if hit)
    n_both = sum(1 for hit in zip(over_strategy, over_oracle, strict=True) if hit[0] and hit[1])
    rate = float(n_both) / float(n_oracle) if n_oracle else 0.0
    return CaptureReport(n_oracle, n_strategy, n_both, n_strategy - n_both, rate)


def peak_return_from_wealth_giveback(terminal_return: float, wealth_giveback: float) -> float:
    """Return-space peak from terminal return and wealth-ratio giveback (W_peak-W_terminal)/W_peak."""
    denom = max(1.0 - float(wealth_giveback), 1e-12)
    return float((1.0 + float(terminal_return)) / denom - 1.0)


def peak_return_from_return_space_giveback(terminal_return: float, giveback: float) -> float:
    """Return-space peak when giveback is peak_return - terminal_return (windows.parquet convention)."""
    return float(terminal_return) + float(giveback)


def return_space_gap(oracle: Sequence[float], terminal: Sequence[float], peak: Sequence[float]) -> GapReport:
    n = len(oracle)
    gaps = [(float(o) - float(p), float(p) - float(t), float(o) - float(t)) for o, t, p in zip(oracle, terminal, peak, strict=True)]
    mean_selection_timing = round(sum(item[0] for item in gaps) / float(n), 12)
    mean_giveback = round(sum(item[1] for item in gaps) / float(n), 12)
    mean_total = round(sum(item[2] for item in gaps) / float(n), 12)
    return GapReport(mean_selection_timing, mean_giveback, mean_total)


def label_comparator_compliance(
    *, effective_gross_max: float, gross_violation_count: int, champion_gross_max: float
) -> str:
    if int(gross_violation_count) > 0 or float(effective_gross_max) > float(champion_gross_max) + 1e-9:
        return "NON_COMPLIANT_REFERENCE"
    return "COMPARABLE"


def population_counts_match(
    *, eligible: int, evaluated: int, router_reset: int, controller: int, shadow: int, horizon: int
) -> bool:
    if eligible <= 0:
        return False
    return (
        evaluated == eligible
        and router_reset == eligible
        and controller == eligible * horizon
        and shadow == eligible * horizon * 4
    )
