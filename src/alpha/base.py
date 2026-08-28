from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import polars as pl

from src.features.regime import RegimeSnapshot
from src.universe.tournament import TournamentRules


@dataclass(frozen=True)
class DecisionContext:
    decision_date: date
    regime: RegimeSnapshot | None
    capital: float
    held: Mapping[str, float]
    rules: TournamentRules


class AlphaModel(Protocol):
    name: str

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]: ...
