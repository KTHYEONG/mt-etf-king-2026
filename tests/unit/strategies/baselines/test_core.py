"""Pin BuyAndHoldBaseline behavior ahead of the P3 factory move."""

from __future__ import annotations

from datetime import date

import polars as pl

from src.alpha.base import DecisionContext
from src.strategies.baselines.core import BuyAndHoldBaseline, make_baseline_buy_hold


def _context() -> DecisionContext:
    return DecisionContext(decision_date=date(2026, 1, 2), regime=None, capital=1.0e9, held={}, rules=None)


def test_buy_and_hold_scores_single_ticker() -> None:
    model = BuyAndHoldBaseline(ticker="069500", name="baseline.buy_hold")
    snapshot = pl.DataFrame({"ticker": ["069500", "123456"]})

    assert model.score(snapshot, _context()) == {"069500": 1.0}
    assert model.name == "baseline.buy_hold"


def test_make_baseline_buy_hold_uses_kodex200() -> None:
    model = make_baseline_buy_hold()

    assert isinstance(model, BuyAndHoldBaseline)
    assert model.name == "baseline.buy_hold"
    assert model.score(pl.DataFrame({"ticker": ["069500"]}), _context()) == {"069500": 1.0}
