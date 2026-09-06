# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

from src.tournament.objective_impl import (
    ChampionshipAdoptionResult,
    ChampionshipObjectiveConfig,
    ChampionshipTailReport,
    FieldRelativeReport,
    championship_tail_report,
    evaluate_championship_adoption,
    field_relative_report,
    paired_scenario_delta_ci,
)


from src.tournament.champion_eval import is_promotable as _is_promotable_ref

_CHAMPION_PROMOTABLE_REF = _is_promotable_ref


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
