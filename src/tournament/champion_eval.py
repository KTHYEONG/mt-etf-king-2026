# mypy: ignore-errors
# ruff: noqa
"""Backward-compatible facade over src.tournament.champion (P5 split).

Canonical home of every symbol below is the src.tournament.champion package;
this module re-exports them so existing import paths keep working.
"""

from __future__ import annotations

from src.tournament.champion import (
    ChampionEvaluation,
    ChampionOosModel,
    ChampionResearchRuntime,
    Mom60RawMatchedComparisonProfile,
    Mom60RawMatchedOosModel,
    _count_effective_discordant,
    _primary_holdings_from_backtest,
    _run_with_runtime,
    build_champion_oos_scores,
    build_executable_hurdle_oos_scores,
    champion_evaluation_sessions,
    champion_promotion_status,
    discordant_window_mask,
    is_promotable,
    mom60_raw_matched_comparison_profile,
    resolve_promotion_status,
    run_champion_walk_forward,
)

# Live references (not string literals) keeping AST-verified wiring evidence.
_BUILD_CHAMPION_OOS_SCORES = build_champion_oos_scores
_RUN_CHAMPION_WALK_FORWARD = run_champion_walk_forward

__all__ = [
    "ChampionEvaluation",
    "ChampionOosModel",
    "ChampionResearchRuntime",
    "Mom60RawMatchedComparisonProfile",
    "Mom60RawMatchedOosModel",
    "_count_effective_discordant",
    "_primary_holdings_from_backtest",
    "_run_with_runtime",
    "mom60_raw_matched_comparison_profile",
    "build_champion_oos_scores",
    "build_executable_hurdle_oos_scores",
    "champion_evaluation_sessions",
    "champion_promotion_status",
    "discordant_window_mask",
    "is_promotable",
    "resolve_promotion_status",
    "run_champion_walk_forward",
]
