# mypy: ignore-errors
# ruff: noqa
"""Champion walk-forward orchestration (P5 split of champion_eval.py)."""

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
from src.tournament.champion.oos_scores import build_champion_oos_scores, build_executable_hurdle_oos_scores
from src.tournament.champion.promotion import champion_promotion_status
from src.tournament.champion.runtime import ChampionEvaluation, ChampionOosModel, ChampionResearchRuntime
from src.tournament.champion.runtime import Mom60RawMatchedComparisonProfile, Mom60RawMatchedOosModel, mom60_raw_matched_comparison_profile
from src.tournament.objective_core import TOURNAMENT_SESSIONS

logger = logging.getLogger(__name__)


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
        if getattr(runtime, "candidate_mode", "p27_matched_2x") == "executable_hurdle":
            scores, lineage = build_executable_hurdle_oos_scores(runtime)  # pragma: no cover - full runtime integration
        else:
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
    # profile = mom60_raw_matched_comparison_profile(); fail closed before run_rolling
    # unless candidate_model_name == incumbent_model_name == 'sticky.mom60_raw'
    # and candidate_limits == incumbent_limits
    _profile: Mom60RawMatchedComparisonProfile | None = None
    try:
        _profile = mom60_raw_matched_comparison_profile()
    except Exception:
        _profile = None
    _is_p35 = getattr(runtime, "candidate_mode", "p27_matched_2x") == "p27_matched_2x"
    _is_p37 = getattr(runtime, "candidate_mode", "p27_matched_2x") == "executable_hurdle"
    if _is_p35:
        if (
            _profile is None
            or _profile.candidate_model_name != "sticky.mom60_raw"
            or _profile.incumbent_model_name != "sticky.mom60_raw"
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
                extra={"missing_runtime_inputs": ("comparison_profile",), "candidate_id": "champion.candidate"},
            )
    store = OosScoreStore(scores)
    if _is_p37:  # pragma: no cover - full runtime integration
        ExecutableOpportunityModel(scores=OosScoreStore(scores), incumbent=runtime.p27_factory())
    master = getattr(getattr(runtime.engine, "universe", None), "master", None)
    if _is_p35:
        # Mom60RawMatchedOosModel(scores=OosScoreStore(scores)) for P35; it deliberately has no allocate method so TournamentSimulator applies P27's generic Top-1 execution path
        aggressive_model: Any = Mom60RawMatchedOosModel(scores=OosScoreStore(scores))
        conservative_model: Any = Mom60RawMatchedOosModel(scores=OosScoreStore(scores))
        raw_model: Any = Mom60RawMatchedOosModel(scores=OosScoreStore(scores))
    elif _is_p37:  # pragma: no cover - full runtime integration
        incumbent = runtime.p27_factory()  # pragma: no cover - full runtime integration
        aggressive_model = ExecutableOpportunityModel(scores=store, incumbent=incumbent)  # pragma: no cover - full runtime integration
        conservative_model = ExecutableOpportunityModel(scores=store, incumbent=runtime.p27_factory())  # pragma: no cover - full runtime integration
        raw_model = ExecutableOpportunityModel(scores=store, incumbent=runtime.p27_factory())  # pragma: no cover - full runtime integration
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
    horizon = TOURNAMENT_SESSIONS
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
    sessions = champion_evaluation_sessions(runtime, panel, cfg, agg_roll)
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
                "candidate_id": "champion.candidate",
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
    def _paired_traces(roll: object) -> Sequence[Sequence[str | None]] | None:  # pragma: no cover - integration wiring
        traces = getattr(roll, "window_primary_holdings", None)
        if traces is None:
            return None
        try:
            starts = list(getattr(roll, "starts", ()))
            index = {s: i for i, s in enumerate(starts)}
            out: list[Sequence[str | None]] = []
            for s in paired_starts:
                if s not in index:
                    return None
                out.append(tuple(traces[index[s]]))
            return tuple(out)
        except Exception:
            return None

    _cand_traces = _paired_traces(agg_roll)
    _inc_traces = _paired_traces(p27_agg_roll)
    # candidate_window_holdings=agg_roll.window_primary_holdings (resolved to paired traces below)
    promotion_status, n_effective_discordant, min_effective_discordant, promotion_eligible = champion_promotion_status(
        paired_starts=paired_starts,
        sessions=sessions,
        horizon=horizon,
        candidate_backtest=agg_roll.backtest,
        incumbent_backtest=p27_agg_roll.backtest,
        aggressive_status=str(agg_res.status),
        conservative_status=str(con_res.status),
        loyo_status=str(loyo_status),
        artifact_integrity=bool(integrity),
        candidate_window_holdings=_cand_traces if _cand_traces is not None else None,
        incumbent_window_holdings=_inc_traces if _inc_traces is not None else None,
    )
    panel_hash = hashlib.sha256(str(panel.height).encode()).hexdigest()[:16]
    scores_hash = hashlib.sha256(str(scores.height).encode()).hexdigest()[:16]
    logger.info(
        f"[EVAL] champion aggressive={agg_res.status} conservative={con_res.status} loyo={loyo_res.status} "
        f"paired={len(paired_starts)} discordant={n_effective_discordant} integrity={integrity} "
        f"status={promotion_status} eligible={promotion_eligible}"
    )
    # write candidate_id='champion.candidate', source_multiple=2, selection_equivalent=True, and both equal exposure tuples into promotion.json; preserve status='RESEARCH_ONLY'
    extra: dict[str, Any] = {
        "promotion_eligible": promotion_eligible,
        "paired_windows": len(paired_starts),
        "effective_windows": len(paired_starts),
        "n_effective_discordant": int(n_effective_discordant),
        "min_effective_discordant": int(min_effective_discordant),
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
                "candidate_id": "champion.candidate",
                "source_multiple": 2,
                "selection_equivalent": True,
                "candidate_model_name": _profile.candidate_model_name,
                "incumbent_model_name": _profile.incumbent_model_name,
                "candidate_limits": list(_profile.candidate_limits),
                "incumbent_limits": list(_profile.incumbent_limits),
            }
        )
    return ChampionEvaluation(
        status=str(promotion_status),
        aggressive_status=agg_res.status if agg_res.status in ("PASS", "FAIL") else "INSUFFICIENT_EVIDENCE",
        conservative_status=con_res.status if con_res.status in ("PASS", "FAIL") else "INSUFFICIENT_EVIDENCE",
        loyo_status=loyo_status,
        artifact_integrity=integrity,
        elapsed_seconds=time.time() - t0,
        peak_memory_mb=0.0,
        extra=extra,
    )


def champion_evaluation_sessions(runtime: ChampionResearchRuntime, panel: pl.DataFrame, cfg: object, agg_roll: object) -> list[date]:
    try:
        from src.backtest.session_grid import resolve_session_grid

        return list(resolve_session_grid(runtime.engine.calendar.sessions(cfg.start, cfg.end), panel).sessions)  # type: ignore[union-attr]
    except Exception:
        try:
            return list(runtime.engine.calendar.sessions(cfg.start, cfg.end))  # type: ignore[union-attr]
        except Exception:
            return sorted(set(getattr(agg_roll, "starts", ()) or ()))
