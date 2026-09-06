"""Pin ConvexImpulseModel semantic identity (P2) ahead of the P3 factory move."""

from __future__ import annotations

from datetime import date

import polars as pl

from src.alpha.base import DecisionContext
from src.strategies.convex_impulse import ConvexImpulseModel
from src.strategies.ids import CONVEX_LOTTERY_IMPULSE


def test_convex_impulse_default_name_is_semantic_id() -> None:
    assert ConvexImpulseModel().name == CONVEX_LOTTERY_IMPULSE
    assert ConvexImpulseModel().name == "convex.lottery_impulse"


def test_convex_impulse_score_empty_snapshot_is_cash() -> None:
    from src.portfolio.intent import CASH_INTENT

    model = ConvexImpulseModel()
    context = DecisionContext(decision_date=date(2026, 1, 2), regime=None, capital=1.0e9, held={}, rules=None)

    assert model.score(pl.DataFrame({"ticker": []}), context) == CASH_INTENT
