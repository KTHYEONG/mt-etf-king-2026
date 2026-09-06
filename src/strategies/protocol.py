"""Static typing contract for strategy factories (P2 semantic-ID unification)."""

from __future__ import annotations

from typing import Protocol

import polars as pl

from src.alpha.base import DecisionContext


class StrategyProtocol(Protocol):
    """Structural type for every value in the strategy registry."""

    name: str

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]: ...
