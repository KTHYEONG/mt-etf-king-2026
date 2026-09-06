# mypy: ignore-errors
# ruff: noqa
"""Backward-compatible facade over split sticky model submodules."""

from __future__ import annotations

from src.strategies.sticky.model_config import (
    DEFAULT_EXCLUDE_NAME_TOKENS,
    StickyLeaderConfig,
    blend_rank_scores,
    cross_section_percentile_ranks,
    logger,
    momentum_horizon,
    name_excluded,
)
from src.strategies.sticky.model_runner import StickyLeaderModel
from src.strategies.sticky.model_scores import apply_sticky_leader, collapse_plus2_by_family, filter_plus2_scores

__all__ = [
    "DEFAULT_EXCLUDE_NAME_TOKENS",
    "StickyLeaderConfig",
    "StickyLeaderModel",
    "apply_sticky_leader",
    "blend_rank_scores",
    "collapse_plus2_by_family",
    "cross_section_percentile_ranks",
    "filter_plus2_scores",
    "logger",
    "momentum_horizon",
    "name_excluded",
]
