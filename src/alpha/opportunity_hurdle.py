# mypy: ignore-errors
"""OOF-calibrated executable opportunity hurdle (P37)."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

import polars as pl

from src.alpha.base import DecisionContext
from src.portfolio.intent import HOLD_INTENT, PortfolioIntent


def _championship_gain(ret: float, objective) -> float:
    grades = objective.grade([float(ret)])
    idx = int(next(iter(grades)))
    gains = tuple(objective.label_gain)
    return float(gains[idx]) if 0 <= idx < len(gains) else 0.0


def calibrate_activation_threshold(
    margins: Sequence[float],
    candidate_returns: Sequence[float],
    incumbent_returns: Sequence[float],
    *,
    objective,
    ruin_threshold: float,
    max_ruin_probability: float,
    min_activations: int,
) -> float | None:
    m = [float(v) for v in margins]
    cr = [float(v) for v in candidate_returns]
    ir = [float(v) for v in incumbent_returns]
    if not m or len(m) != len(cr) or len(m) != len(ir):
        return None
    for v in (*m, *cr, *ir):
        if not math.isfinite(v):
            return None
    if not math.isfinite(float(ruin_threshold)) or not math.isfinite(float(max_ruin_probability)):
        return None
    if int(min_activations) <= 0:
        return None
    cands = sorted({v for v in m if math.isfinite(v)}, reverse=True)
    best: float | None = None
    best_uplift = float("-inf")
    best_ruin = float("inf")
    for t in cands:
        idx = [i for i, v in enumerate(m) if v >= t]
        if len(idx) < int(min_activations):
            continue
        ruin_n = sum(1 for i in idx if float(cr[i]) <= float(ruin_threshold))
        ruin = ruin_n / len(idx)
        if ruin > float(max_ruin_probability) + 1e-12:
            continue
        cand_score = sum(_championship_gain(cr[i], objective) for i in idx) / len(idx)
        inc_score = sum(_championship_gain(ir[i], objective) for i in idx) / len(idx)
        if not cand_score >= inc_score - 1e-12:
            continue
        uplift = cand_score - inc_score
        if uplift > best_uplift + 1e-12 or (abs(uplift - best_uplift) <= 1e-12 and (ruin < best_ruin - 1e-12 or (abs(ruin - best_ruin) <= 1e-12 and (best is None or t > best)))):
            best = float(t)
            best_uplift = float(uplift)
            best_ruin = float(ruin)
    return best


@dataclass(frozen=True)
class ExecutableHurdleArtifact:
    threshold_models: tuple[object, ...]
    thresholds: tuple[float, ...]
    weights: tuple[float, ...]
    activation_threshold: float | None
    trained_through: date
    backend: Literal["lightgbm"]
    artifact_hash: str


def fit_executable_hurdle(
    train: pl.DataFrame,
    calibration: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    objective,
    seed: int,
    min_activations: int,
) -> ExecutableHurdleArtifact:
    import numpy as np

    from src.alpha.champion_ranker import ChampionTrainingError

    cols = tuple(feature_columns)
    if not cols:
        raise ChampionTrainingError("artifact-integrity failure: empty feature columns")
    if "label_return" not in train.columns:
        raise ChampionTrainingError("artifact-integrity failure: missing label_return")
    thresholds = tuple(float(v) for v in objective.thresholds)
    models: list[object] = []
    for thr in thresholds:
        try:
            sub = train.select([*cols, "label_return"]).drop_nulls()
            xs = np.ascontiguousarray(sub.select(list(cols)).to_numpy(), dtype=np.float64)
            rets = np.asarray(sub.get_column("label_return").to_numpy(), dtype=np.float64)
            if xs.shape[0] == 0:
                raise ChampionTrainingError("artifact-integrity failure: no finite training rows")
            if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(rets)):
                raise ChampionTrainingError("artifact-integrity failure: non-finite training values")
            ys = np.ascontiguousarray((rets > float(thr)).astype(np.int32))
            import lightgbm as lgb

            ds = lgb.Dataset(xs, label=ys)
            params = {
                "objective": "binary", "verbosity": -1, "seed": int(seed),
                "deterministic": True, "force_col_wise": True, "num_threads": 1,
                "num_leaves": 8, "max_depth": 4, "min_data_in_leaf": 100,
            }
            booster = lgb.train(params, ds, num_boost_round=50)
            probe = booster.predict(xs[: min(1, xs.shape[0])])
            for v in list(probe):
                if not math.isfinite(float(v)):
                    raise ChampionTrainingError("artifact-integrity failure: non-finite prediction")
            models.append(booster)
        except ChampionTrainingError:
            raise
        except Exception as exc:
            raise ChampionTrainingError(f"artifact-integrity failure: hurdle fit failed: {exc}") from exc
    # OOF calibration: predict calibration rows with the same fold-local models.
    margins: list[float] = []
    cand_rets: list[float] = []
    inc_rets: list[float] = []
    try:
        if calibration.height > 0 and all(c in calibration.columns for c in cols):
            xc = np.ascontiguousarray(calibration.select(list(cols)).to_numpy(), dtype=np.float64)
            probs = [np.asarray(m.predict(xc), dtype=np.float64) for m in models]
            wts = [float(w) for w in objective.weights]
            direct = sum(w * p for w, p in zip(wts, probs, strict=True))
            has_cand = "label_return" in calibration.columns
            has_inc = "incumbent_return" in calibration.columns
            incumbent_by_date: dict[object, float] = {}
            if "incumbent_ticker" in calibration.columns and "decision_date" in calibration.columns:
                tickers = calibration.get_column("source_ticker").to_list() if "source_ticker" in calibration.columns else []
                dates = calibration.get_column("decision_date").to_list()
                for d in set(dates):
                    idxs = [j for j, dd in enumerate(dates) if dd == d]
                    selected = next((j for j in idxs if str(tickers[j]) == str(calibration.filter(pl.col("decision_date") == d).get_column("incumbent_ticker")[0])), idxs[0])
                    incumbent_by_date[d] = float(direct[selected])
            for i in range(calibration.height):
                d = calibration.get_column("decision_date")[i] if "decision_date" in calibration.columns else None
                margins.append(float(direct[i] - incumbent_by_date[d]) if d in incumbent_by_date else float(direct[i]))
                cand_rets.append(float(calibration.get_column("label_return").to_list()[i]) if has_cand else 0.0)
                inc_rets.append(float(calibration.get_column("incumbent_return").to_list()[i]) if has_inc else 0.0)
    except Exception as exc:
        raise ChampionTrainingError(f"artifact-integrity failure: calibration failed: {exc}") from exc
    activation: float | None = None
    if margins and cand_rets and inc_rets:
        try:
            from src.core.config import load_config

            gates = load_config("gates")
            gates_section = gates.get("gates", {}) if isinstance(gates, dict) else {}
            ruin_thr = float(gates_section.get("g2a_ruin_threshold", -0.25))
            max_ruin = float(gates_section.get("g2a_max_prob", 0.05))
        except Exception:
            ruin_thr, max_ruin = -0.25, 0.05
        activation = calibrate_activation_threshold(margins, cand_rets, inc_rets, objective=objective, ruin_threshold=ruin_thr, max_ruin_probability=max_ruin, min_activations=int(min_activations))
    through = train.get_column("decision_date").max() if "decision_date" in train.columns and train.height else date(2026, 1, 2)
    digest = hashlib.sha256(f"{cols}{thresholds}{seed}{train.height}".encode()).hexdigest()[:16]
    import lightgbm as lgb

    getattr(lgb, "__version__", "unknown")
    return ExecutableHurdleArtifact(
        threshold_models=tuple(models),
        thresholds=thresholds,
        weights=tuple(float(w) for w in objective.weights),
        activation_threshold=activation,
        trained_through=through,  # type: ignore[arg-type]
        backend="lightgbm",
        artifact_hash=digest,
    )


class ExecutableOpportunityModel:
    def __init__(self, *, scores, incumbent) -> None:
        self._scores = scores
        self._incumbent = incumbent

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float] | PortfolioIntent:
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
        try:
            decided, activate = self._scores.decision_for(context.decision_date, eligible)
        except Exception:
            return self._incumbent.score(snapshot, context)
        if not activate:
            return self._incumbent.score(snapshot, context)
        # Missing incumbent score disables activation.
        try:
            inc_scores = self._incumbent.score(snapshot, context)
        except Exception:
            return HOLD_INTENT
        if isinstance(inc_scores, dict) and not inc_scores:
            return HOLD_INTENT
        out: dict[str, float] = {}
        for k, v in decided.items():
            fv = float(v)
            if not math.isfinite(fv):
                continue
            out[str(k)] = fv
        if not out:
            return self._incumbent.score(snapshot, context)
        return out


__all__ = [
    "ExecutableHurdleArtifact",
    "ExecutableOpportunityModel",
    "calibrate_activation_threshold",
    "fit_executable_hurdle",
]
