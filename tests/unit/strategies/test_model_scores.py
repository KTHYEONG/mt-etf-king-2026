"""Facade fidelity for the sticky model scores split (P5)."""

from __future__ import annotations


def test_model_scores_canonical_home() -> None:
    import src.strategies.sticky.model as facade
    from src.strategies.sticky.model_scores import apply_sticky_leader, filter_plus2_scores

    assert facade.apply_sticky_leader is apply_sticky_leader
    assert facade.filter_plus2_scores is filter_plus2_scores
    assert apply_sticky_leader.__module__ == "src.strategies.sticky.model_scores"
