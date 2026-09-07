"""Canonical decision population for championship feasibility research (Phase 0)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import polars as pl

from src.backtest.session_grid import resolve_session_grid


class WindowPolicy(StrEnum):
    CALENDAR_NAIVE = "calendar_naive"
    PANEL_NAIVE = "panel_naive"
    EXECUTABLE = "executable"


class PopulationError(ValueError):
    """Fail-closed error for invalid decision-population inputs."""


@dataclass(frozen=True, slots=True)
class DecisionWindow:
    window_idx: int
    decision_date: date
    entry_date: date
    exit_date: date
    return_session_ids: tuple[date, ...]


@dataclass(frozen=True, slots=True)
class ExclusionRecord:
    date: date
    policy: WindowPolicy
    reason: str


@dataclass(frozen=True, slots=True)
class DecisionPopulation:
    policy: WindowPolicy
    horizon: int
    sessions: tuple[date, ...]
    phantom: tuple[date, ...]
    windows: tuple[DecisionWindow, ...]
    exclusions: tuple[ExclusionRecord, ...]
    n_windows: int


def build_decision_population(
    *,
    calendar_sessions: Sequence[date],
    panel: pl.DataFrame,
    horizon: int,
    policy: WindowPolicy,
) -> DecisionPopulation:
    if horizon < 1 or len(calendar_sessions) == 0 or panel.height == 0:
        raise PopulationError(f"fail-closed: horizon={horizon} calendar={len(calendar_sessions)} panel_height={panel.height}")
    if policy is WindowPolicy.CALENDAR_NAIVE:
        sessions = tuple(calendar_sessions)
        phantom = resolve_session_grid(calendar_sessions, panel).phantom
        windows = tuple(
            DecisionWindow(i, sessions[i], sessions[i], sessions[i + horizon - 1], tuple(sessions[i : i + horizon]))
            for i in range(len(sessions) - horizon + 1)
        )
        return DecisionPopulation(policy, horizon, sessions, phantom, windows, (), len(windows))
    grid = resolve_session_grid(calendar_sessions, panel)
    sessions = grid.sessions
    phantom = grid.phantom
    phantom_exclusions = tuple(ExclusionRecord(day, policy, "PHANTOM_SESSION") for day in phantom)
    if policy is WindowPolicy.PANEL_NAIVE:
        windows = tuple(
            DecisionWindow(i, sessions[i], sessions[i], sessions[i + horizon - 1], tuple(sessions[i : i + horizon]))
            for i in range(len(sessions) - horizon + 1)
        )
        return DecisionPopulation(policy, horizon, sessions, phantom, windows, phantom_exclusions, len(windows))
    windows = tuple(
        DecisionWindow(i, sessions[i], sessions[i + 1], sessions[i + 1 + horizon], tuple(sessions[i + 1 : i + 1 + horizon]))
        for i in range(len(sessions) - horizon - 1)
    )
    exclusions = (
        *phantom_exclusions,
        ExclusionRecord(sessions[0], policy, "NO_PRIOR_DECISION"),
        ExclusionRecord(sessions[-1], policy, "NO_EXIT_OPEN"),
    )
    return DecisionPopulation(policy, horizon, sessions, phantom, windows, exclusions, len(windows))
