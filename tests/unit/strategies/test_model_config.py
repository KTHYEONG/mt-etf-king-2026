"""Facade fidelity for the sticky model config split (P5)."""

from __future__ import annotations


def test_model_config_canonical_home() -> None:
    import src.strategies.sticky.model as facade
    from src.strategies.sticky.model_config import DEFAULT_EXCLUDE_NAME_TOKENS, StickyLeaderConfig

    assert facade.StickyLeaderConfig is StickyLeaderConfig
    assert facade.DEFAULT_EXCLUDE_NAME_TOKENS is DEFAULT_EXCLUDE_NAME_TOKENS
    assert StickyLeaderConfig.__module__ == "src.strategies.sticky.model_config"
