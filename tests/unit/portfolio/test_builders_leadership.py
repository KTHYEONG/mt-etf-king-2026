"""Canonical-home checks for the portfolio strategy builders (P5 layering)."""

from __future__ import annotations


def test_builders_leadership_canonical_home() -> None:
    from src.portfolio.builders_leadership import (
        make_portfolio_leadership_confidence,
        make_portfolio_leadership_policy,
    )
    from src.strategies.registry import STRATEGIES

    assert make_portfolio_leadership_policy.__module__ == "src.portfolio.builders_leadership"
    assert "portfolio.leadership_policy" in STRATEGIES
    assert "portfolio.leadership_confidence" in STRATEGIES
