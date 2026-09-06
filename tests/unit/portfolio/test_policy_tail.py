"""Facade fidelity for the src portfolio policy tail split (P5)."""

from __future__ import annotations


def test_policy_tail_canonical_home() -> None:
    import src.portfolio.policy as facade
    from src.portfolio.policy_tail import PortfolioPolicyTailMixin

    assert facade.PortfolioPolicyTailMixin is PortfolioPolicyTailMixin
    assert PortfolioPolicyTailMixin.__module__ == "src.portfolio.policy_tail"
    assert issubclass(facade.PortfolioPolicy, PortfolioPolicyTailMixin)
