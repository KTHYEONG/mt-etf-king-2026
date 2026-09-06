# mypy: ignore-errors
# ruff: noqa
"""Tournament objective package (P5 split of objective_impl.py + objective_core.py).

Config-backed gate constants remain assigned in src.tournament.objective_core
(single source of truth); this package holds the gate logic.
"""

from __future__ import annotations

from src.tournament.objective.adoption import evaluate_championship_adoption
from src.tournament.objective.core import ObjectiveGateConfig
from src.tournament.objective.core import ObjectiveGateResult
from src.tournament.objective.core import P15AdoptionReport
from src.tournament.objective.core import P16AdoptionReport
from src.tournament.objective.core import evaluate_objective_gates
from src.tournament.objective.core import evaluate_p15_adoption_report
from src.tournament.objective.core import evaluate_p16_adoption_report
from src.tournament.objective.core import paired_tail_delta_ci
from src.tournament.objective.reports import ChampionshipAdoptionResult
from src.tournament.objective.reports import ChampionshipObjectiveConfig
from src.tournament.objective.reports import ChampionshipTailReport
from src.tournament.objective.reports import FieldRelativeReport
from src.tournament.objective.reports import GROSS_METRIC_UNAVAILABLE
from src.tournament.objective.reports import championship_tail_report
from src.tournament.objective.reports import field_relative_report
from src.tournament.objective.reports import paired_scenario_delta_ci

__all__ = [
    "ChampionshipAdoptionResult",
    "ChampionshipObjectiveConfig",
    "ChampionshipTailReport",
    "FieldRelativeReport",
    "GROSS_METRIC_UNAVAILABLE",
    "ObjectiveGateConfig",
    "ObjectiveGateResult",
    "P15AdoptionReport",
    "P16AdoptionReport",
    "championship_tail_report",
    "evaluate_championship_adoption",
    "evaluate_objective_gates",
    "evaluate_p15_adoption_report",
    "evaluate_p16_adoption_report",
    "field_relative_report",
    "paired_scenario_delta_ci",
    "paired_tail_delta_ci",
]
