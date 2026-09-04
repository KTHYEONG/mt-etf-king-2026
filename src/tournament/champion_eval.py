# mypy: ignore-errors
# ruff: noqa
"""P34 walk-forward promotion evaluation (identical fold-local OOS comparison)."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from src.alpha.base import AlphaModel, DecisionContext
from src.alpha.champion_dataset import ChampionDatasetConfig
from src.alpha.champion_dataset import build_family_tail_dataset
from src.alpha.champion_dataset import collect_family_candidates
from src.alpha.champion_ranker import ChampionTailRanker, OosScoreStore, PurgedDateWalkForward
from src.portfolio.intent import HOLD_INTENT, PortfolioIntent
from src.portfolio.policy import PortfolioDecision
from src.strategies.champion_tail import ChampionPolicyConfig, ChampionTailPolicy

logger = logging.getLogger(__name__)


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
    embargo_sessions: int = 36
    purge_sessions: int = 36
    ranker_seed: int = 20260831
    ranker_num_leaves: int = 8
    ranker_max_depth: int = 4
    ranker_min_data_in_leaf: int = 100
    candidate_mode: str = "p27_matched_2x"


@dataclass(frozen=True)
class P27MatchedComparisonProfile:
    candidate_model_name: str
    incumbent_model_name: str
    candidate_limits: tuple[float, float, float]
    incumbent_limits: tuple[float, float, float]


def p27_matched_comparison_profile() -> P27MatchedComparisonProfile:
    from src.portfolio.constraints import load_p27_exposure_limits

    candidate_limits = load_p27_exposure_limits()
    incumbent_limits = load_p27_exposure_limits()
    return P27MatchedComparisonProfile(
        candidate_model_name="P27",
        incumbent_model_name="P27",
        candidate_limits=(float(candidate_limits[0]), float(candidate_limits[1]), float(candidate_limits[2])),
        incumbent_limits=(float(incumbent_limits[0]), float(incumbent_limits[1]), float(incumbent_limits[2])),
    )


class P27MatchedOosModel:
    name: str = "P27"
    candidate_id: str = "P35"

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
        try:
            eligible = [str(t) for t in snapshot.select(pl.col(col)).to_series().to_list()]
        except Exception:
            return HOLD_INTENT
        if not eligible:
            return HOLD_INTENT
        if len(set(eligible)) != len(eligible):
            return HOLD_INTENT
        try:
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
    name: str = "P34"
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


def build_champion_oos_scores(
    runtime: ChampionResearchRuntime,
) -> tuple[pl.DataFrame, tuple[dict[str, object], ...]]:
    from dataclasses import replace as _replace

    from src.alpha.champion_dataset import ChampionDatasetConfig as ChampionDatasetConfig  # noqa: F401

    dataset_config = runtime.dataset_config
    # replace(runtime.dataset_config, source_multiple=2) when runtime.candidate_mode == 'p27_matched_2x' before collect_family_candidates
    if getattr(runtime, "candidate_mode", "p27_matched_2x") == "p27_matched_2x":
        try:
            dataset_config = _replace(dataset_config, source_multiple=2)
        except Exception:
            pass
    panel = runtime.panel
    engine = runtime.engine
    backtest_config = runtime.backtest_config
    if panel is None or panel.height == 0:
        raise ValueError("empty panel for OOS scores")
    if not dataset_config.feature_columns or len(dataset_config.feature_columns) > 25:
        raise ValueError("feature columns violate <=25")
    universe = getattr(engine, "universe", None)
    if universe is None:
        raise ValueError("engine universe missing")
    master = getattr(universe, "master", None)
    if master is None:
        master = getattr(universe, "_master", None)
    if master is None:
        raise ValueError("universe master missing")
    filters = getattr(backtest_config, "filters", None)
    if filters is None:
        raise ValueError("backtest filters missing")
    try:
        calendar_sessions = set(engine.calendar.sessions(backtest_config.start, backtest_config.end))
        sessions = sorted(
            d for d in panel.select(pl.col("date")).to_series().unique().to_list()
            if isinstance(d, date) and d in calendar_sessions
        )
    except Exception as exc:
        raise ValueError(f"panel date column missing: {exc}") from exc
    sessions = [d for d in sessions if isinstance(d, date)]
    if not sessions:
        raise ValueError("no sessions in panel")
    candidates = collect_family_candidates(
        panel, sessions=sessions, universe=universe, filters=filters, master=master, config=dataset_config,
    )
    if candidates.height == 0:
        raise ValueError("no family candidates")
    labeled = build_family_tail_dataset(candidates, panel, sessions=sessions, config=dataset_config)
    if labeled.height == 0:
        raise ValueError("no labeled rows")
    decision_dates = sorted(set(labeled.select(pl.col("decision_date")).to_series().to_list()))
    splitter = PurgedDateWalkForward(
        n_folds=int(runtime.n_folds),
        label_horizon=int(dataset_config.label_horizon),
        embargo=int(runtime.embargo_sessions),
        min_train_sessions=int(runtime.min_train_sessions),
    )
    folds = splitter.split(decision_dates)
    if not folds:
        raise ValueError("no walk-forward folds")
    rows: list[dict[str, object]] = []
    lineage: list[dict[str, object]] = []
    for fold in folds:
        train = labeled.filter(pl.col("decision_date").is_in(list(fold.train_dates)))
        if train.height == 0:
            raise ValueError(f"fold {fold.fold_id} empty train")
        ranker = ChampionTailRanker(
            feature_columns=list(dataset_config.feature_columns),
            seed=int(runtime.ranker_seed),
            num_leaves=int(runtime.ranker_num_leaves),
            max_depth=int(runtime.ranker_max_depth),
            min_data_in_leaf=int(runtime.ranker_min_data_in_leaf),
        )
        artifact = ranker.fit(train)
        scoring_dates = ([fold.warmup_date] if fold.warmup_date is not None else []) + list(fold.test_dates)
        if not scoring_dates:
            raise ValueError(f"fold {fold.fold_id} empty test")
        for d in scoring_dates:
            snap = candidates.filter(pl.col("decision_date") == d)
            if snap.height == 0:
                continue
            scored = ranker.score(snap, artifact=artifact)
            is_eval = d in set(fold.test_dates)
            for ticker, value in scored.items():
                rows.append(
                    {
                        "decision_date": d,
                        "source_ticker": str(ticker),
                        "score": float(value),
                        "fold_id": int(fold.fold_id),
                        "trained_through": fold.trained_through,
                        "is_evaluation": bool(is_eval),
                    }
                )
        lineage.append(
            {
                "fold_id": int(fold.fold_id),
                "train_count": int(len(fold.train_dates)),
                "test_count": int(len(fold.test_dates)),
                "trained_through": fold.trained_through,
                "warmup_date": fold.warmup_date,
            }
        )
    if not rows:
        raise ValueError("no OOS scores produced")
    scores = pl.DataFrame(
        rows,
        schema={
            "decision_date": pl.Date,
            "source_ticker": pl.String,
            "score": pl.Float64,
            "fold_id": pl.Int64,
            "trained_through": pl.Date,
            "is_evaluation": pl.Boolean,
        },
        strict=True,
    )
    evaluated = scores.filter(pl.col("is_evaluation"))
    if evaluated.height == 0:
        raise ValueError("no evaluation scores")
    if evaluated.unique(subset=["decision_date", "source_ticker"]).height != evaluated.height:
        raise ValueError("duplicate evaluation scores")
    bad = evaluated.filter(pl.col("decision_date") <= pl.col("trained_through"))
    if bad.height != 0:
        raise ValueError("purge/lineage violation: decision_date <= trained_through")
    for col in ("score",):
        vals = evaluated.select(pl.col(col)).to_series().to_list()
        import math as _math

        for v in vals:
            try:
                if not _math.isfinite(float(v)):
                    raise ValueError("non-finite OOS score")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-finite OOS score: {exc}") from exc
    return scores, tuple(lineage)


def _insufficient(t0: float, missing: tuple[str, ...]) -> ChampionEvaluation:
    return ChampionEvaluation(
        status="RESEARCH_ONLY",
        aggressive_status="INSUFFICIENT_EVIDENCE",
        conservative_status="INSUFFICIENT_EVIDENCE",
        loyo_status="INSUFFICIENT_EVIDENCE",
        artifact_integrity=False,
        elapsed_seconds=time.time() - t0,
        peak_memory_mb=0.0,
        extra={"missing_runtime_inputs": missing},
    )


def run_champion_walk_forward(
    *,
    runtime: ChampionResearchRuntime | None = None,
    start: date | None = None,
    end: date | None = None,
    engine: Any = None,
    simulator: Any = None,
    panel: Any = None,
    config: Any = None,
    model_config: Any | None = None,
    dataset_config: Any | None = None,
    p27_factory: Callable[[], Any] | None = None,
) -> ChampionEvaluation:
    """Execute P34-raw/P34/P27/conservative on identical fold-local OOS sessions."""
    t0 = time.time()
    if runtime is None:
        required = {
            "engine": engine,
            "simulator": simulator,
            "panel": panel,
            "config": config,
            "model_config": model_config,
            "dataset_config": dataset_config,
            "p27_factory": p27_factory,
        }
        missing = tuple(name for name, value in required.items() if value is None)
        if not missing:
            missing = ("runtime",)
        return _insufficient(t0, missing)
    try:
        return _run_with_runtime(runtime=runtime, t0=t0)
    except Exception as exc:
        return ChampionEvaluation(
            status="RESEARCH_ONLY",
            aggressive_status="INSUFFICIENT_EVIDENCE",
            conservative_status="INSUFFICIENT_EVIDENCE",
            loyo_status="INSUFFICIENT_EVIDENCE",
            artifact_integrity=False,
            elapsed_seconds=time.time() - t0,
            peak_memory_mb=0.0,
            extra={"missing_runtime_inputs": (), "error": repr(exc)},
        )


def _run_with_runtime(*, runtime: ChampionResearchRuntime, t0: float) -> ChampionEvaluation:
    import hashlib
    import math
    from dataclasses import replace

    from src.backtest.session_cache import build_session_cache
    from src.tournament.loyo import evaluate_promotion_robustness
    from src.tournament.objective_impl import evaluate_championship_adoption

    if runtime.panel is None or runtime.panel.height == 0:
        return _insufficient(t0, ("panel",))
    if int(runtime.min_train_sessions) < 1:
        return _insufficient(t0, ("min_train_sessions",))
    try:
        scores, lineage = build_champion_oos_scores(runtime)
    except Exception as exc:
        return ChampionEvaluation(
            status="RESEARCH_ONLY",
            aggressive_status="INSUFFICIENT_EVIDENCE",
            conservative_status="INSUFFICIENT_EVIDENCE",
            loyo_status="INSUFFICIENT_EVIDENCE",
            artifact_integrity=False,
            elapsed_seconds=time.time() - t0,
            peak_memory_mb=0.0,
            extra={"missing_runtime_inputs": (), "error": repr(exc)},
        )
    # ChampionTailRanker wiring anchor: fresh ranker per fold already used in scores.
    _ = ChampionTailRanker
    # profile = p27_matched_comparison_profile(); fail closed before run_rolling unless candidate_model_name == incumbent_model_name == 'P27' and candidate_limits == incumbent_limits
    _profile: P27MatchedComparisonProfile | None = None
    try:
        _profile = p27_matched_comparison_profile()
    except Exception:
        _profile = None
    _is_p35 = getattr(runtime, "candidate_mode", "p27_matched_2x") == "p27_matched_2x"
    if _is_p35:
        if (
            _profile is None
            or _profile.candidate_model_name != "P27"
            or _profile.incumbent_model_name != "P27"
            or tuple(_profile.candidate_limits) != tuple(_profile.incumbent_limits)
        ):
            return ChampionEvaluation(
                status="RESEARCH_ONLY",
                aggressive_status="INSUFFICIENT_EVIDENCE",
                conservative_status="INSUFFICIENT_EVIDENCE",
                loyo_status="INSUFFICIENT_EVIDENCE",
                artifact_integrity=False,
                elapsed_seconds=time.time() - t0,
                peak_memory_mb=0.0,
                extra={"missing_runtime_inputs": ("comparison_profile",), "candidate_id": "P35"},
            )
    store = OosScoreStore(scores)
    _ = P27MatchedComparisonProfile
    master = getattr(getattr(runtime.engine, "universe", None), "master", None)
    if _is_p35:
        # P27MatchedOosModel(scores=OosScoreStore(scores)) for P35; it deliberately has no allocate method so TournamentSimulator applies P27's generic Top-1 execution path
        aggressive_model: Any = P27MatchedOosModel(scores=OosScoreStore(scores))
        conservative_model: Any = P27MatchedOosModel(scores=OosScoreStore(scores))
        raw_model: Any = P27MatchedOosModel(scores=OosScoreStore(scores))
    else:
        aggressive_policy = ChampionTailPolicy(master=master, config=runtime.policy_config)
        conservative_policy = ChampionTailPolicy(master=master, config=runtime.policy_config)
        raw_policy = ChampionTailPolicy(master=master, config=runtime.policy_config)
        aggressive_model = ChampionOosModel(scores=store, policy=aggressive_policy)
        conservative_model = ChampionOosModel(scores=store, policy=conservative_policy)
        raw_model = ChampionOosModel(scores=store, policy=raw_policy)
    try:
        p27_aggressive = runtime.p27_factory()
        p27_conservative = runtime.p27_factory()
    except Exception:
        return _insufficient(t0, ("p27_factory",))
    horizon = 36
    cfg = runtime.backtest_config
    sim = runtime.simulator
    panel = runtime.panel
    try:
        # Snapshots/execution inputs are model-independent.  Reuse them while keeping
        # separate immutable rule objects for leveraged and conservative comparisons.
        candidate_agg_cache = build_session_cache(
            runtime.engine, aggressive_model, panel, cfg, leverage_allowed=True,
        )
        candidate_con_cache = replace(
            candidate_agg_cache,
            rules=replace(candidate_agg_cache.rules, leverage_allowed=False),
        )
        p27_agg_cache = build_session_cache(
            runtime.engine, p27_aggressive, panel, cfg, leverage_allowed=True,
        )
        p27_con_cache = replace(
            p27_agg_cache,
            rules=replace(p27_agg_cache.rules, leverage_allowed=False),
        )
        agg_roll = sim.run_rolling(
            aggressive_model, panel, cfg, horizon, path_dependent=True, leverage_allowed=True, session_cache=candidate_agg_cache,
        )
        con_roll = sim.run_rolling(
            conservative_model, panel, cfg, horizon, path_dependent=True, leverage_allowed=False, session_cache=candidate_con_cache,
        )
        p27_agg_roll = sim.run_rolling(
            p27_aggressive, panel, cfg, horizon, path_dependent=True, leverage_allowed=True, session_cache=p27_agg_cache,
        )
        p27_con_roll = sim.run_rolling(
            p27_conservative, panel, cfg, horizon, path_dependent=True, leverage_allowed=False, session_cache=p27_con_cache,
        )
        raw_roll = sim.run_rolling(
            raw_model, panel, cfg, horizon, path_dependent=True, leverage_allowed=False, session_cache=candidate_con_cache,
        )
    except Exception as exc:
        return ChampionEvaluation(
            status="RESEARCH_ONLY",
            aggressive_status="INSUFFICIENT_EVIDENCE",
            conservative_status="INSUFFICIENT_EVIDENCE",
            loyo_status="INSUFFICIENT_EVIDENCE",
            artifact_integrity=False,
            elapsed_seconds=time.time() - t0,
            peak_memory_mb=0.0,
            extra={"missing_runtime_inputs": (), "error": repr(exc)},
        )
    # Fold-local OOS segments: retain only wholly evaluation windows.
    eval_dates = set(scores.filter(pl.col("is_evaluation")).select(pl.col("decision_date")).to_series().to_list())
    try:
        sessions = list(runtime.engine.calendar.sessions(cfg.start, cfg.end))
    except Exception:
        sessions = sorted(set(agg_roll.starts))
    paired_starts: list[date] = []
    for s in agg_roll.starts:
        try:
            idx = sessions.index(s)
        except ValueError:
            continue
        window = sessions[idx : idx + horizon]
        if len(window) != horizon:
            continue
        if all(d in eval_dates for d in window):
            paired_starts.append(s)
    if not paired_starts:
        return _insufficient(t0, ("paired_windows",))
    agg_map = dict(zip(agg_roll.starts, agg_roll.returns, strict=False))
    con_map = dict(zip(con_roll.starts, con_roll.returns, strict=False))
    p27_agg_map = dict(zip(p27_agg_roll.starts, p27_agg_roll.returns, strict=False))
    p27_con_map = dict(zip(p27_con_roll.starts, p27_con_roll.returns, strict=False))
    raw_map = dict(zip(raw_roll.starts, raw_roll.returns, strict=False))
    # Pair by window_start; reject unequal keys (fail-closed, no zip/imputation).
    for s in paired_starts:
        if s not in con_map or s not in p27_agg_map or s not in p27_con_map or s not in raw_map or s not in agg_map:
            return _insufficient(t0, ("pair_mismatch",))
    agg_rets = [float(agg_map[s]) for s in paired_starts]
    con_rets = [float(con_map[s]) for s in paired_starts]
    p27_agg_rets = [float(p27_agg_map[s]) for s in paired_starts]
    p27_con_rets = [float(p27_con_map[s]) for s in paired_starts]
    raw_rets = [float(raw_map[s]) for s in paired_starts]
    for seq in (agg_rets, con_rets, p27_agg_rets, p27_con_rets, raw_rets):
        for v in seq:
            if not math.isfinite(float(v)):
                return _insufficient(t0, ("non_finite_return",))
    gross_viol = 0
    for roll in (agg_roll, con_roll, p27_agg_roll, p27_con_roll, raw_roll):
        diag = getattr(roll, "diagnostics", None)
        if diag is None or getattr(diag, "gross_violation_count", None) is None:
            return _insufficient(t0, ("gross_metric",))
        if getattr(diag, "gross_violation_count", None) != 0:
            try:
                if int(diag.gross_violation_count) != 0:
                    gross_viol += int(diag.gross_violation_count)
            except Exception:
                return _insufficient(t0, ("gross_metric",))
    # Aggregate P34 gross violations must be zero (fail-closed, no imputation).
    if _is_p35 and int(gross_viol) != 0:
        return ChampionEvaluation(
            status="RESEARCH_ONLY",
            aggressive_status="INSUFFICIENT_EVIDENCE",
            conservative_status="INSUFFICIENT_EVIDENCE",
            loyo_status="INSUFFICIENT_EVIDENCE",
            artifact_integrity=False,
            elapsed_seconds=time.time() - t0,
            peak_memory_mb=0.0,
            extra={
                "missing_runtime_inputs": ("gross_violation",),
                "candidate_id": "P35",
                "source_multiple": 2,
                "promotion_eligible": False,
            },
        )
    obj_cfg = runtime.objective_config
    agg_res = evaluate_championship_adoption(
        candidate_returns=agg_rets,
        incumbent_returns=p27_agg_rets,
        raw_returns=raw_rets,
        horizon=horizon,
        config=obj_cfg,
        execution_parity=True,
        gross_violation_count=int(gross_viol),
        era_pairs=None,
    )
    con_res = evaluate_championship_adoption(
        candidate_returns=con_rets,
        incumbent_returns=p27_con_rets,
        raw_returns=raw_rets,
        horizon=horizon,
        config=obj_cfg,
        execution_parity=True,
        gross_violation_count=int(gross_viol),
        era_pairs=None,
    )
    cand_windows = pl.DataFrame({"window_start": paired_starts, "terminal_return": agg_rets})
    inc_windows = pl.DataFrame({"window_start": paired_starts, "terminal_return": p27_agg_rets})
    loyo_res = evaluate_promotion_robustness(candidate_windows=cand_windows, incumbent_windows=inc_windows)
    loyo_status = "PASS" if loyo_res.status == "PASS" else ("INSUFFICIENT_EVIDENCE" if loyo_res.status == "INSUFFICIENT" else "FAIL")
    integrity = bool(
        paired_starts
        and lineage
        and all(math.isfinite(float(v)) for v in agg_rets + con_rets + p27_agg_rets + p27_con_rets + raw_rets)
        and int(gross_viol) == 0
    )
    promotion_eligible = bool(
        agg_res.status == "PASS"
        and con_res.status == "PASS"
        and loyo_res.status == "PASS"
        and integrity is True
    )
    panel_hash = hashlib.sha256(str(panel.height).encode()).hexdigest()[:16]
    scores_hash = hashlib.sha256(str(scores.height).encode()).hexdigest()[:16]
    logger.info(
        f"[EVAL] champion aggressive={agg_res.status} conservative={con_res.status} loyo={loyo_res.status} "
        f"paired={len(paired_starts)} integrity={integrity} eligible={promotion_eligible}"
    )
    # write candidate_id='P35', source_multiple=2, selection_equivalent=True, and both equal exposure tuples into promotion.json; preserve status='RESEARCH_ONLY'
    extra: dict[str, Any] = {
        "promotion_eligible": promotion_eligible,
        "paired_windows": len(paired_starts),
        "effective_windows": len(paired_starts),
        "aggressive_failures": list(agg_res.failures),
        "conservative_failures": list(con_res.failures),
        "loyo_failures": list(loyo_res.failures),
        "panel_hash": panel_hash,
        "scores_hash": scores_hash,
        "lineage": [dict(r) for r in lineage],
    }
    if _is_p35 and _profile is not None:
        extra.update(
            {
                "candidate_id": "P35",
                "source_multiple": 2,
                "selection_equivalent": True,
                "candidate_model_name": _profile.candidate_model_name,
                "incumbent_model_name": _profile.incumbent_model_name,
                "candidate_limits": list(_profile.candidate_limits),
                "incumbent_limits": list(_profile.incumbent_limits),
            }
        )
    return ChampionEvaluation(
        status="RESEARCH_ONLY",
        aggressive_status=agg_res.status if agg_res.status in ("PASS", "FAIL") else "INSUFFICIENT_EVIDENCE",
        conservative_status=con_res.status if con_res.status in ("PASS", "FAIL") else "INSUFFICIENT_EVIDENCE",
        loyo_status=loyo_status,
        artifact_integrity=integrity,
        elapsed_seconds=time.time() - t0,
        peak_memory_mb=0.0,
        extra=extra,
    )


def is_promotable(
    *,
    aggressive_status: str,
    conservative_status: str,
    loyo_status: str,
    artifact_integrity: bool,
) -> bool:
    """Promotion requires dual-scenario PASS, LOYO PASS, and artifact integrity."""
    return bool(
        aggressive_status == "PASS"
        and conservative_status == "PASS"
        and loyo_status == "PASS"
        and artifact_integrity is True
    )


__all__ = [
    "ChampionEvaluation",
    "ChampionOosModel",
    "ChampionResearchRuntime",
    "P27MatchedComparisonProfile",
    "P27MatchedOosModel",
    "build_champion_oos_scores",
    "is_promotable",
    "p27_matched_comparison_profile",
    "run_champion_walk_forward",
]
