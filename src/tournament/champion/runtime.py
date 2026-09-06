# mypy: ignore-errors
# ruff: noqa
"""Champion research runtime models (P5 split of champion_eval.py)."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from src.alpha.base import AlphaModel, DecisionContext
from src.alpha.champion_dataset import ChampionDatasetConfig
from src.alpha.champion_dataset import build_family_tail_dataset
from src.alpha.champion_dataset import collect_direct_vehicle_candidates
from src.alpha.champion_dataset import collect_family_candidates
from src.alpha.champion_ranker import ChampionTailRanker, ChampionTrainingError, OosScoreStore, PurgedDateWalkForward, TailGradeObjective, is_valid_champion_artifact
from src.alpha.executable_labels import ExecutableLabelConfig, build_executable_vehicle_labels
from src.alpha.opportunity_hurdle import ExecutableOpportunityModel, fit_executable_hurdle
from src.portfolio.intent import HOLD_INTENT, PortfolioIntent
from src.portfolio.policy import PortfolioDecision
from src.strategies.champion_tail import ChampionPolicyConfig, ChampionTailPolicy
from src.tournament.objective_core import TOURNAMENT_SESSIONS


@dataclass
class ChampionEvaluation:
    status: str = "RESEARCH_ONLY"
    aggressive_status: str = "FAIL"
    conservative_status: str = "FAIL"
    loyo_status: str = "FAIL"
    artifact_integrity: bool = False
    elapsed_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def write(self, path: str | Path) -> Path:
        import json

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": self.status,
            "aggressive_status": self.aggressive_status,
            "conservative_status": self.conservative_status,
            "loyo_status": self.loyo_status,
            "artifact_integrity": self.artifact_integrity,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            **self.extra,
        }
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(dest)
        return dest


@dataclass(frozen=True)
class ChampionResearchRuntime:
    engine: Any
    simulator: Any
    panel: pl.DataFrame
    backtest_config: Any
    dataset_config: ChampionDatasetConfig
    objective_config: Any
    policy_config: ChampionPolicyConfig
    p27_factory: Callable[[], AlphaModel]
    min_train_sessions: int
    n_folds: int = 3
    embargo_sessions: int = TOURNAMENT_SESSIONS
    purge_sessions: int = TOURNAMENT_SESSIONS
    ranker_seed: int = 20260831
    ranker_num_leaves: int = 8
    ranker_max_depth: int = 4
    ranker_min_data_in_leaf: int = 100
    candidate_mode: str = "p27_matched_2x"


@dataclass(frozen=True)
class Mom60RawMatchedComparisonProfile:
    candidate_model_name: str
    incumbent_model_name: str
    candidate_limits: tuple[float, float, float]
    incumbent_limits: tuple[float, float, float]


def mom60_raw_matched_comparison_profile() -> Mom60RawMatchedComparisonProfile:
    from src.portfolio.constraints import load_p27_exposure_limits

    candidate_limits = load_p27_exposure_limits()
    incumbent_limits = load_p27_exposure_limits()
    return Mom60RawMatchedComparisonProfile(
        candidate_model_name="sticky.mom60_raw",
        incumbent_model_name="sticky.mom60_raw",
        candidate_limits=(float(candidate_limits[0]), float(candidate_limits[1]), float(candidate_limits[2])),
        incumbent_limits=(float(incumbent_limits[0]), float(incumbent_limits[1]), float(incumbent_limits[2])),
    )


class Mom60RawMatchedOosModel:
    name: str = "sticky.mom60_raw"
    candidate_id: str = "champion.candidate"

    def __init__(self, *, scores: OosScoreStore) -> None:
        self._scores = scores

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float] | PortfolioIntent:
        import math as _math

        if snapshot is None or getattr(snapshot, "height", 0) == 0:
            return HOLD_INTENT
        cols = list(getattr(snapshot, "columns", []))
        if "ticker" in cols:
            col = "ticker"
        elif "source_ticker" in cols:
            col = "source_ticker"
        else:
            return HOLD_INTENT
        try:  # pragma: no cover - runtime wiring fallback
            eligible = [str(t) for t in snapshot.select(pl.col(col)).to_series().to_list()]
        except Exception:
            return HOLD_INTENT
        if not eligible:
            return HOLD_INTENT
        if len(set(eligible)) != len(eligible):
            return HOLD_INTENT
        try:  # pragma: no cover - runtime wiring fallback
            out = self._scores.scores_for(context.decision_date, eligible)
        except Exception:
            return HOLD_INTENT
        if not out:
            return HOLD_INTENT
        for v in out.values():
            try:
                if not _math.isfinite(float(v)):
                    return HOLD_INTENT
            except Exception:
                return HOLD_INTENT
        return {str(k): float(v) for k, v in out.items()}


class ChampionOosModel:
    name: str = "champion.oos_model"
    scores_path_independent: bool = False
    path_dependent: bool = True

    def __init__(self, *, scores: OosScoreStore, policy: ChampionTailPolicy) -> None:
        self._scores = scores
        self._policy = policy

    @property
    def policy(self) -> ChampionTailPolicy:
        return self._policy

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float] | PortfolioIntent:
        if snapshot is None or snapshot.height == 0:
            return HOLD_INTENT
        if "ticker" in snapshot.columns:
            col = "ticker"
        elif "source_ticker" in snapshot.columns:
            col = "source_ticker"
        else:
            return HOLD_INTENT
        try:
            eligible = [str(t) for t in snapshot.select(pl.col(col)).to_series().to_list()]
        except Exception:
            return HOLD_INTENT
        if not eligible:
            return HOLD_INTENT
        try:
            return self._scores.scores_for(context.decision_date, eligible)
        except Exception:
            return HOLD_INTENT

    def allocate(self, scores: Mapping[str, float], **kwargs: object) -> PortfolioDecision:
        return self._policy.allocate(scores, **kwargs)
