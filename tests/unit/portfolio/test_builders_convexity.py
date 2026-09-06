"""Canonical-home checks for the portfolio strategy builders (P5 layering)."""

from __future__ import annotations


def test_builders_convexity_canonical_home() -> None:
    from src.portfolio.builders_convexity import make_portfolio_convexity_hold, make_portfolio_lottery_exposure
    from src.strategies.registry import STRATEGIES

    assert make_portfolio_lottery_exposure.__module__ == "src.portfolio.builders_convexity"
    assert "portfolio.lottery_exposure" in STRATEGIES
    assert "portfolio.convexity_hold" in STRATEGIES
