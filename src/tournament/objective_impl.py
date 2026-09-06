# mypy: ignore-errors
# ruff: noqa
"""Backward-compatible facade over src.tournament.objective (P5 split).

Canonical home of every symbol below is the src.tournament.objective package
(moved from objective_impl.py); this module re-exports them so existing import
paths keep working.
"""

from __future__ import annotations

from src.tournament.objective import ChampionshipAdoptionResult
from src.tournament.objective import ChampionshipObjectiveConfig
from src.tournament.objective import ChampionshipTailReport
from src.tournament.objective import FieldRelativeReport
from src.tournament.objective import GROSS_METRIC_UNAVAILABLE
from src.tournament.objective import ObjectiveGateConfig
from src.tournament.objective import ObjectiveGateResult
from src.tournament.objective import P15AdoptionReport
from src.tournament.objective import P16AdoptionReport
from src.tournament.objective import championship_tail_report
from src.tournament.objective import evaluate_championship_adoption
from src.tournament.objective import evaluate_objective_gates
from src.tournament.objective import evaluate_p15_adoption_report
from src.tournament.objective import evaluate_p16_adoption_report
from src.tournament.objective import field_relative_report
from src.tournament.objective import paired_scenario_delta_ci
from src.tournament.objective import paired_tail_delta_ci
from src.tournament.objective_core import _dist_exceedance

__all__ = [
    "ObjectiveGateConfig",
    "ObjectiveGateResult",
    "paired_tail_delta_ci",
    "evaluate_objective_gates",
    "_dist_exceedance",
    "P15AdoptionReport",
    "evaluate_p15_adoption_report",
    "P16AdoptionReport",
    "evaluate_p16_adoption_report",
    "ChampionshipObjectiveConfig",
    "ChampionshipTailReport",
    "ChampionshipAdoptionResult",
    "GROSS_METRIC_UNAVAILABLE",
    "championship_tail_report",
    "paired_scenario_delta_ci",
    "FieldRelativeReport",
    "field_relative_report",
    "evaluate_championship_adoption",
]
