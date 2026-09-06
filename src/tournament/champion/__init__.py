# mypy: ignore-errors
# ruff: noqa
"""Champion research package (P5 split of src/tournament/champion_eval.py)."""

from __future__ import annotations

from src.tournament.champion.oos_scores import build_champion_oos_scores
from src.tournament.champion.oos_scores import build_executable_hurdle_oos_scores
from src.tournament.champion.promotion import _primary_holdings_from_backtest
from src.tournament.champion.promotion import _count_effective_discordant
from src.tournament.champion.promotion import _primary_holdings_from_backtest
from src.tournament.champion.promotion import champion_promotion_status
from src.tournament.champion.promotion import discordant_window_mask
from src.tournament.champion.promotion import is_promotable
from src.tournament.champion.promotion import resolve_promotion_status
from src.tournament.champion.runtime import ChampionEvaluation
from src.tournament.champion.runtime import ChampionOosModel
from src.tournament.champion.runtime import ChampionResearchRuntime
from src.tournament.champion.runtime import Mom60RawMatchedComparisonProfile
from src.tournament.champion.runtime import Mom60RawMatchedOosModel
from src.tournament.champion.runtime import mom60_raw_matched_comparison_profile
from src.tournament.champion.walkforward import _insufficient
from src.tournament.champion.walkforward import _run_with_runtime
from src.tournament.champion.walkforward import _run_with_runtime
from src.tournament.champion.walkforward import champion_evaluation_sessions
from src.tournament.champion.walkforward import run_champion_walk_forward

__all__ = [
    "ChampionEvaluation",
    "ChampionOosModel",
    "ChampionResearchRuntime",
    "Mom60RawMatchedComparisonProfile",
    "Mom60RawMatchedOosModel",
    "mom60_raw_matched_comparison_profile",
    "build_champion_oos_scores",
    "build_executable_hurdle_oos_scores",
    "champion_evaluation_sessions",
    "_count_effective_discordant",
    "_primary_holdings_from_backtest",
    "_run_with_runtime",
    "champion_promotion_status",
    "discordant_window_mask",
    "is_promotable",
    "resolve_promotion_status",
    "run_champion_walk_forward",
]
