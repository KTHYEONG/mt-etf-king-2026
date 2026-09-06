"""Canonical-home checks for the split-fill sticky model (P5 layering)."""

from __future__ import annotations


def test_split_fill_model_canonical_home() -> None:
    from src.strategies.registry import STRATEGIES
    from src.strategies.sticky.split_fill_model import SplitFillStickyModel, families_from_snapshot

    assert SplitFillStickyModel.__module__ == "src.strategies.sticky.split_fill_model"
    assert families_from_snapshot.__module__ == "src.strategies.sticky.split_fill_model"
    assert "sticky.split_fill_lock" in STRATEGIES
