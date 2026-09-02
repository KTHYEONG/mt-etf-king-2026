# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import polars as pl

from src.alpha.base import DecisionContext


class BuyAndHoldBaseline:
    def __init__(self, ticker: str, name: str = "B0") -> None:
        self.ticker = ticker
        self.name = name

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]:
        if snapshot.height > 0 and "ticker" in snapshot.columns:
            tickers = snapshot.select(pl.col("ticker")).to_series().to_list()
            if self.ticker in tickers:
                return {self.ticker: 1.0}
        return {self.ticker: 1.0} if self.ticker else {}


def make_baseline_buy_hold() -> BuyAndHoldBaseline:
    return BuyAndHoldBaseline(ticker="069500", name="baseline.buy_hold")


def make_alpha_sector_leadership() -> object:
    from src.alpha.baselines import _make_m07

    return _make_m07()


__all__ = [
    "BuyAndHoldBaseline",
    "make_baseline_buy_hold",
    "make_alpha_sector_leadership",
]
