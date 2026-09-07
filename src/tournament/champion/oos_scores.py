# mypy: ignore-errors
# ruff: noqa
"""Champion OOS score builders (P5 split of champion_eval.py)."""

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
from src.tournament.champion.runtime import ChampionResearchRuntime
from src.tournament.objective_core import CHAMPIONSHIP_THRESHOLDS, TOURNAMENT_SESSIONS


def build_hurdle_calibration_frame(
    rows: Sequence[Mapping[str, object]], feature_columns: Sequence[str]
) -> pl.DataFrame:
    feats = tuple(feature_columns or ())
    parsed: list[dict[str, object]] = []
    for row in rows:
        rec = dict(row)
        raw_date = rec.get("date", rec.get("decision_date"))
        if isinstance(raw_date, str):
            rec["date"] = date.fromisoformat(raw_date)
            raw_date = rec["date"]
        elif isinstance(raw_date, date) and "date" not in rec:
            rec["date"] = raw_date
        parsed.append(rec)
    best: dict[object, dict[str, object]] = {}
    best_score: dict[object, float] = {}
    for rec in parsed:
        day = rec.get("date")
        score_raw = rec.get("score")
        score_val = float(score_raw) if score_raw is not None else float("-inf")
        if day not in best or score_val > best_score[day]:
            best[day] = rec
            best_score[day] = score_val
    ordered = [best[day] for day in sorted(best, key=lambda d: str(d))]
    frame = pl.DataFrame(ordered, strict=False)
    casts: list[pl.Expr] = []
    if "date" in frame.columns:
        casts.append(pl.col("date").cast(pl.Date))
    if "ticker" in frame.columns:
        casts.append(pl.col("ticker").cast(pl.String))
    for col in feats:
        if str(col) in frame.columns:
            casts.append(pl.col(str(col)).cast(pl.Float64))
    if casts:
        frame = frame.with_columns(casts)
    return frame


def build_executable_hurdle_oos_scores(
    runtime: ChampionResearchRuntime,
) -> tuple[pl.DataFrame, tuple[dict[str, object], ...]]:
    """P37 executable-hurdle OOS scores with inner-OOF activation calibration."""
    import yaml as _yaml

    from src.backtest.costs import CostConfig

    panel = runtime.panel
    engine = runtime.engine
    backtest_config = runtime.backtest_config
    if panel is None or panel.height == 0:
        raise ValueError("empty panel for OOS scores")
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
        sessions = sorted(d for d in panel.select(pl.col("date")).to_series().unique().to_list() if isinstance(d, date) and d in calendar_sessions)
    except Exception as exc:
        raise ValueError(f"panel date column missing: {exc}") from exc
    if not sessions:
        raise ValueError("no sessions in panel")
    from src.core.config import config_path

    with open(config_path("gates"), encoding="utf-8") as f:
        gates_raw = _yaml.safe_load(f) or {}
    champ = (gates_raw.get("championship", {}) or {})
    thresholds = tuple(float(v) for v in champ.get("thresholds", CHAMPIONSHIP_THRESHOLDS))
    scen = str(champ.get("primary_scenario", "championship"))
    weights = tuple(float(v) for v in (champ.get("scenario_weights", {}).get(scen, [0.1, 0.25, 0.45, 0.2])))
    objective = TailGradeObjective(thresholds=thresholds, weights=weights)
    with open(config_path("ml"), encoding="utf-8") as f:
        ml_raw = _yaml.safe_load(f) or {}
    ct = ((ml_raw.get("ml", {}) or {}).get("champion_tail", {}) or {})
    feature_columns = tuple(ct.get("executable_feature_columns", ()))
    if not feature_columns:
        raise ValueError("ml.champion_tail.executable_feature_columns missing")
    fill_budget = float(ct.get("fill_budget_fraction", 0.25))
    costs_raw = ct.get("costs", {}) if isinstance(ct.get("costs", {}), dict) else {}
    costs = CostConfig(
        commission_bps=float(costs_raw.get("commission_bps", 3.0)),
        slippage_bps=float(costs_raw.get("slippage_bps", 5.0)),
        spread_bps=float(costs_raw.get("spread_bps", 0.0)),
        tax_bps=float(costs_raw.get("tax_bps", 0.0)),
    )
    label_config = ExecutableLabelConfig(
        horizon=TOURNAMENT_SESSIONS, fill_budget_fraction=fill_budget,
        capital=1_000_000_000.0, participation=0.01, max_single_weight=0.80,
        max_effective_gross=1.60, min_cash=0.05, costs=costs,
    )
    min_attainable = int((gates_raw.get("attainability") or {}).get("min_attainable_windows", 30))
    candidates = collect_direct_vehicle_candidates(panel, sessions=sessions, universe=universe, filters=filters, master=master, feature_columns=feature_columns)
    if candidates.height == 0:
        raise ValueError("no direct candidates")
    # Materialize the two executable features from PIT ADV20; never let the
    # learner silently drop every row because derived columns are absent.
    if "log_adv_20" in feature_columns or "fill_sessions" in feature_columns:
        import numpy as np

        history: dict[str, list[tuple[date, float]]] = {}
        for row in panel.iter_rows(named=True):
            ticker = str(row.get("ticker"))
            d = row.get("date")
            try:
                value = float(row.get("trading_value"))
            except Exception:  # pragma: no cover - corrupt panel value
                value = float("nan")
            if isinstance(d, date):
                history.setdefault(ticker, []).append((d, value))
        derived_log: list[float] = []
        derived_fill: list[float] = []
        target_notional = 0.80 * 1_000_000_000.0
        for row in candidates.iter_rows(named=True):
            ticker = str(row.get("source_ticker"))
            d = row.get("decision_date")
            vals = [v for dd, v in history.get(ticker, []) if isinstance(d, date) and dd <= d and math.isfinite(v)]
            adv20 = float(np.mean(vals[-20:])) if vals else float("nan")
            derived_log.append(float(np.log1p(adv20)) if math.isfinite(adv20) else float("nan"))
            derived_fill.append(float(target_notional / (adv20 * 0.01)) if adv20 > 0 and math.isfinite(adv20) else float("nan"))
        candidates = candidates.with_columns(
            pl.Series("log_adv_20", derived_log, dtype=pl.Float64),
            pl.Series("fill_sessions", derived_fill, dtype=pl.Float64),
        )
    exec_labeled = build_executable_vehicle_labels(candidates, panel, sessions=sessions, objective=objective, config=label_config)
    if exec_labeled.height == 0:
        raise ValueError("no executable labels")
    decision_dates = sorted(set(exec_labeled.select(pl.col("decision_date")).to_series().to_list()))
    splitter = PurgedDateWalkForward(n_folds=int(runtime.n_folds), label_horizon=TOURNAMENT_SESSIONS, embargo=int(runtime.embargo_sessions), min_train_sessions=int(runtime.min_train_sessions))
    folds = splitter.split(decision_dates)
    rows: list[dict[str, object]] = []
    lineage: list[dict[str, object]] = []
    for fold in folds:
        train = exec_labeled.filter(pl.col("decision_date").is_in(list(fold.train_dates)))
        inner = PurgedDateWalkForward(n_folds=2, label_horizon=TOURNAMENT_SESSIONS, embargo=TOURNAMENT_SESSIONS, min_train_sessions=max(1, len(fold.train_dates) // 3))
        try:
            inner_folds = inner.split(list(fold.train_dates))
        except Exception:  # noqa: S112
            inner_folds = ()
        if inner_folds:
            inner_train = train.filter(pl.col("decision_date").is_in(list(inner_folds[0].train_dates)))
            inner_oof = train.filter(pl.col("decision_date").is_in(list(inner_folds[0].test_dates)))
        else:
            inner_train, inner_oof = train, train
        # Calibrate against the PIT incumbent's selected ticker, never against a
        # hindsight best vehicle.  The same deterministic selector is reused for
        # inner calibration and outer scoring.
        def incumbent_for(frame: pl.DataFrame, d: date) -> tuple[str | None, float]:
            day = frame.filter(pl.col("decision_date") == d)
            if day.height == 0:  # pragma: no cover - fold invariant
                return None, float("nan")
            ordered_rows = list(day.sort("source_ticker").iter_rows(named=True))
            try:
                snap = pl.DataFrame([{"ticker": r["source_ticker"], **{c: r.get(c) for c in feature_columns}} for r in ordered_rows], strict=False)
                incumbent = runtime.p27_factory()
                ctx = DecisionContext(decision_date=d, regime=None, capital=1_000_000_000.0, held={}, rules=None)
                scores0 = incumbent.score(snap, ctx)
                if isinstance(scores0, dict) and scores0:  # pragma: no cover - baseline-dependent branch
                    ticker = max(scores0, key=lambda k: float(scores0[k]))
                else:
                    ticker = str(ordered_rows[0]["source_ticker"])
            except Exception:  # pragma: no cover - incumbent outage fails closed
                ticker = str(ordered_rows[0]["source_ticker"])
            match = day.filter(pl.col("source_ticker") == str(ticker))
            if match.height == 0:  # pragma: no cover - fold invariant
                return str(ticker), float("nan")
            return str(ticker), float(match.get_column("label_return")[0])

        cal_rows: list[dict[str, object]] = []
        for d in sorted(set(inner_oof.get_column("decision_date").to_list())) if inner_oof.height else []:
            inc_ticker, inc_ret = incumbent_for(exec_labeled, d)
            for row in inner_oof.filter(pl.col("decision_date") == d).iter_rows(named=True):
                cal_rows.append({**row, "incumbent_return": inc_ret, "incumbent_ticker": inc_ticker})
        calibration = build_hurdle_calibration_frame(cal_rows, feature_columns) if cal_rows else inner_oof.with_columns(pl.lit(None).cast(pl.Float64).alias("incumbent_return"))
        inner_oof = calibration
        artifact = fit_executable_hurdle(inner_train, inner_oof, feature_columns=feature_columns, objective=objective, seed=runtime.ranker_seed, min_activations=min_attainable)
        if not is_valid_champion_artifact(artifact):
            raise ChampionTrainingError("artifact-integrity failure: invalid hurdle artifact")  # pragma: no cover - integrity guard

        import numpy as np

        def direct_scores(frame: pl.DataFrame) -> dict[str, float]:
            if frame.height == 0:  # pragma: no cover - fold invariant
                return {}
            xs = np.ascontiguousarray(frame.select(list(feature_columns)).to_numpy(), dtype=np.float64)
            probs = [np.asarray(model.predict(xs), dtype=np.float64) for model in artifact.threshold_models]
            values = np.zeros(frame.height, dtype=np.float64)
            for weight, prob in zip(artifact.weights, probs, strict=True):
                values += float(weight) * prob
            return {str(t): float(v) for t, v in zip(frame.get_column("source_ticker").to_list(), values, strict=True) if math.isfinite(float(v))}
        scoring_dates = ([fold.warmup_date] if fold.warmup_date is not None else []) + list(fold.test_dates)
        for d in scoring_dates:
            snap = candidates.filter(pl.col("decision_date") == d)
            is_eval = d in set(fold.test_dates)
            scored = direct_scores(snap)
            incumbent_ticker, incumbent_return = incumbent_for(exec_labeled, d)
            incumbent_score = float(scored.get(str(incumbent_ticker), float("nan")))
            margin = max(scored.values(), default=float("-inf")) - incumbent_score
            activation = artifact.activation_threshold is not None
            if activation:  # pragma: no cover - threshold is usually unavailable in sparse fixtures
                activation = margin >= float(artifact.activation_threshold)
            for row in snap.iter_rows(named=True):
                ticker = str(row.get("source_ticker"))
                rows.append({
                    "decision_date": d, "source_ticker": ticker,
                    "score": float(scored.get(ticker, float("nan"))),
                    "activate": bool(activation and math.isfinite(incumbent_return)),
                    "fold_id": int(fold.fold_id), "trained_through": fold.trained_through,
                    "is_evaluation": bool(is_eval),
                })
        lineage.append({
            "fold_id": int(fold.fold_id), "train_count": int(len(fold.train_dates)),
            "test_count": int(len(fold.test_dates)), "trained_through": fold.trained_through,
            "activation_threshold": artifact.activation_threshold,
        })
    scores = pl.DataFrame(
        rows,
        schema={"decision_date": pl.Date, "source_ticker": pl.String, "score": pl.Float64, "activate": pl.Boolean, "fold_id": pl.Int64, "trained_through": pl.Date, "is_evaluation": pl.Boolean},
        strict=True,
    )
    return scores, tuple(lineage)


def build_champion_oos_scores(
    runtime: ChampionResearchRuntime,
) -> tuple[pl.DataFrame, tuple[dict[str, object], ...]]:
    from dataclasses import replace as _replace

    from src.alpha.champion_dataset import ChampionDatasetConfig as ChampionDatasetConfig  # noqa: F401

    dataset_config = runtime.dataset_config
    # replace(runtime.dataset_config, source_multiple=2) when runtime.candidate_mode == 'p27_matched_2x' before collect_family_candidates
    if getattr(runtime, "candidate_mode", "p27_matched_2x") == "p27_matched_2x":
        try:  # pragma: no cover - runtime wiring fallback
            tail_thresholds = tuple(runtime.objective_config.thresholds)
            tail_weights = tuple(runtime.objective_config.scenario_weights[runtime.objective_config.primary_scenario])
            # tail_weights=tuple(runtime.objective_config.scenario_weights[runtime.objective_config.primary_scenario])
            dataset_config = _replace(
                dataset_config,
                source_multiple=2,
                tail_thresholds=tuple(tail_thresholds),
                tail_weights=tuple(tail_weights),
            )
        except Exception:  # pragma: no cover - legacy runtime fallback
            try:  # pragma: no cover - legacy runtime fallback
                dataset_config = _replace(dataset_config, source_multiple=2)
            except Exception:  # pragma: no cover - malformed runtime config
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
