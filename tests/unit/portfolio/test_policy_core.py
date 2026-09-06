"""Facade fidelity for the src portfolio policy core split (P5)."""

from __future__ import annotations


def test_policy_core_canonical_home() -> None:
    import src.portfolio.policy as facade
    from src.portfolio.policy_core import PortfolioPolicy

    assert facade.PortfolioPolicy is PortfolioPolicy
    assert PortfolioPolicy.__module__ == "src.portfolio.policy_core"
