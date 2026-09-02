# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

from src.tournament.objective_impl import *  # noqa: F403
from src.tournament.championship import evaluate_championship_adoption

__all__ = ["evaluate_championship_adoption"]
