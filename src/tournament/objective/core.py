# mypy: ignore-errors
# ruff: noqa
"""Core objective gates: canonical home is src.tournament.objective_core.

This module re-exports the gate logic so the package surface is uniform.
Config-backed constants (CHAMPIONSHIP_THRESHOLDS, TOURNAMENT_SESSIONS,
ATTAINABILITY_THRESHOLDS, BOOTSTRAP_EXPECTED_BLOCK) are assigned in
src.tournament.objective_core and imported from there.
"""

from __future__ import annotations

from src.tournament.objective_core import ObjectiveGateConfig
from src.tournament.objective_core import ObjectiveGateResult
from src.tournament.objective_core import P15AdoptionReport
from src.tournament.objective_core import P16AdoptionReport
from src.tournament.objective_core import _dist_exceedance
from src.tournament.objective_core import evaluate_objective_gates
from src.tournament.objective_core import evaluate_p15_adoption_report
from src.tournament.objective_core import evaluate_p16_adoption_report
from src.tournament.objective_core import paired_tail_delta_ci

__all__ = [
    "ObjectiveGateConfig",
    "ObjectiveGateResult",
    "P15AdoptionReport",
    "P16AdoptionReport",
    "_dist_exceedance",
    "evaluate_objective_gates",
    "evaluate_p15_adoption_report",
    "evaluate_p16_adoption_report",
    "paired_tail_delta_ci",
]
