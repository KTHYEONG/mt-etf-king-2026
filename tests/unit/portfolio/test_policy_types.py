"""Facade fidelity for the src portfolio policy split (P5)."""

from __future__ import annotations


def test_policy_types_canonical_home() -> None:
    import src.portfolio.policy as facade
    from src.portfolio.policy_types import PathDependentPolicyError, PortfolioDecision

    assert facade.PathDependentPolicyError is PathDependentPolicyError
    assert facade.PortfolioDecision is PortfolioDecision
    assert PortfolioDecision.__module__ == "src.portfolio.policy_types"
