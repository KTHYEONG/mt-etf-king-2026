"""Facade fidelity for the sticky model runner split (P5)."""

from __future__ import annotations


def test_model_runner_canonical_home() -> None:
    import src.strategies.sticky.model as facade
    from src.strategies.sticky.model_runner import StickyLeaderModel

    assert facade.StickyLeaderModel is StickyLeaderModel
    assert StickyLeaderModel.__module__ == "src.strategies.sticky.model_runner"
