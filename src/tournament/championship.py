# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

from src.tournament.objective_impl import (
    ChampionshipAdoptionResult,
    ChampionshipObjectiveConfig,
    ChampionshipTailReport,
    FieldRelativeReport,
    championship_tail_report,
    evaluate_championship_adoption as _evaluate_championship_adoption,
    field_relative_report as _field_relative_report,
    paired_scenario_delta_ci,
)


def evaluate_championship_adoption(*args, **kwargs):
    return _evaluate_championship_adoption(*args, **kwargs)


def field_relative_report(*args, **kwargs):
    return _field_relative_report(*args, **kwargs)


__all__ = [
    "ChampionshipAdoptionResult",
    "ChampionshipObjectiveConfig",
    "ChampionshipTailReport",
    "FieldRelativeReport",
    "championship_tail_report",
    "evaluate_championship_adoption",
    "field_relative_report",
    "paired_scenario_delta_ci",
]
