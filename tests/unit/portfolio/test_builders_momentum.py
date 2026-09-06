"""Canonical-home checks for the portfolio strategy builders (P5 layering)."""

from __future__ import annotations


def test_builders_momentum_canonical_home() -> None:
    from src.portfolio.builders_momentum import (
        make_portfolio_momentum_confidence,
        make_portfolio_momentum_policy,
        make_portfolio_momentum_vehicle,
    )
    from src.strategies.registry import STRATEGIES

    assert make_portfolio_momentum_policy.__module__ == "src.portfolio.builders_momentum"
    assert "portfolio.momentum_policy" in STRATEGIES
    assert "portfolio.momentum_vehicle" in STRATEGIES
    assert "portfolio.momentum_confidence" in STRATEGIES
