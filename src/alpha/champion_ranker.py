# mypy: ignore-errors
# ruff: noqa: S101
"""Purged date walk-forward splitter, LambdaRank tail ranker, OOS score store."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

import polars as pl


@dataclass(frozen=True)
class DateFold:
    fold_id: int
    train_dates: tuple[date, ...]
    test_dates: tuple[date, ...]
    warmup_date: date | None = None
    trained_through: date | None = None


@dataclass(frozen=True)
class PurgedDateWalkForward:
    n_folds: int
    label_horizon: int
    embargo: int
    min_train_sessions: int

    def split(self, decision_dates: Sequence[date]) -> tuple[DateFold, ...]:
        ordered = sorted(set(decision_dates))
        n = len(ordered)
        if self.n_folds <= 0 or n == 0:
            raise ValueError("no decision dates for walk-forward split")
        gap = int(self.label_horizon) + int(self.embargo)
        if gap < 72:
            # Contract floor: purge >= 36 and embargo >= 36 (gap >= 72).
            gap = 72
        usable = n - int(self.min_train_sessions) - gap
        if usable <= 0:
            raise ValueError("insufficient sessions for purged walk-forward")
        test_size = max(1, usable // int(self.n_folds))
        folds: list[DateFold] = []
        for k in range(int(self.n_folds)):
            train_end_excl = int(self.min_train_sessions) + k * test_size
            test_start = train_end_excl + gap
            test_end = test_start + test_size if k < int(self.n_folds) - 1 else n
            if test_start >= n:
                break
            test_end = min(test_end, n)
            train_dates = tuple(ordered[:train_end_excl])
            test_dates = tuple(ordered[test_start:test_end])
            if not train_dates or not test_dates:
                continue
            warmup = ordered[test_start - 1] if test_start - 1 >= train_end_excl else None
            folds.append(
                DateFold(
                    fold_id=k,
                    train_dates=train_dates,
                    test_dates=test_dates,
                    warmup_date=warmup,
                    trained_through=train_dates[-1],
                )
            )
        return tuple(folds)


@dataclass(frozen=True)
class TailGradeObjective:
    thresholds: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        thr = tuple(float(v) for v in self.thresholds)
        wts = tuple(float(v) for v in self.weights)
        if len(thr) == 0 or len(thr) != len(wts):
            raise ValueError("thresholds/weights must be non-empty and equal length")
        for v in thr:
            if not math.isfinite(v) or v <= 0:
                raise ValueError("thresholds must be positive finite")
        for i in range(1, len(thr)):
            if not thr[i] > thr[i - 1]:
                raise ValueError("thresholds must be strictly ascending")
        for v in wts:
            if not math.isfinite(v) or v < 0:
                raise ValueError("weights must be finite non-negative")
        if sum(wts) <= 0:
            raise ValueError("weights must have positive total")

    def grade(self, returns: Sequence[float]) -> object:
        import numpy as np

        arr = np.asarray(list(returns), dtype=np.float64)
        thr = np.asarray([float(v) for v in self.thresholds], dtype=np.float64)
        grades = np.zeros(arr.shape, dtype=np.int32)
        for t in thr:
            grades += (arr > float(t)).astype(np.int32)
        return np.ascontiguousarray(grades, dtype=np.int32)

    @property
    def label_gain(self) -> tuple[float, ...]:
        cum: list[float] = [0.0]
        total = 0.0
        for w in self.weights:
            total += float(w)
            cum.append(round(total, 12))
        return tuple(cum)


class ChampionTrainingError(RuntimeError):
    pass


@dataclass
class ChampionModelArtifact:
    artifact_id: str
    trained_through: date | None
    feature_columns: tuple[str, ...]
    model: object = None
    feature_config_hash: str = ""
    model_config_hash: str = ""
    panel_hash: str = ""
    lgbm_version: str = ""
    target_column: str = "label_rank"
    backend: Literal["lightgbm"] = "lightgbm"


class ChampionTailRanker:
    """Deterministic CPU LambdaRank grouped by decision_date (shallow capacity)."""

    def __init__(
        self,
        feature_columns: Sequence[str] | None = None,
        *,
        seed: int = 20260831,
        num_leaves: int = 8,
        max_depth: int = 4,
        min_data_in_leaf: int = 100,
        objective: TailGradeObjective | None = None,
    ) -> None:
        cols = tuple(feature_columns or ())
        if len(cols) > 25:
            raise ValueError("feature count exceeds 25")
        if int(num_leaves) > 8 or int(max_depth) > 4 or int(min_data_in_leaf) < 100:
            raise ValueError("model capacity exceeds shallow limits")
        self._features = cols
        self._seed = int(seed)
        self._objective = objective
        self._params = {
            "num_leaves": int(num_leaves),
            "max_depth": int(max_depth),
            "min_data_in_leaf": int(min_data_in_leaf),
        }

    def fit(self, train: pl.DataFrame) -> ChampionModelArtifact:
        target_column = "label_rank"
        has_objective = self._objective is not None
        if has_objective:
            if "label_return" not in train.columns or "decision_date" not in train.columns:
                raise ChampionTrainingError("artifact-integrity failure: missing label/decision_date")
        else:
            if target_column not in train.columns:
                raise ValueError("artifact-integrity failure: missing label_rank target")
            if "label_return" not in train.columns or "decision_date" not in train.columns:
                raise ValueError("artifact-integrity failure: missing label/decision_date")
        cols = list(self._features) if self._features else [c for c in train.columns if c not in ("label_return", "label_rank", "decision_date", "source_ticker", "family_key")]
        missing = [c for c in cols if c not in train.columns]
        if missing:
            if has_objective:
                raise ChampionTrainingError(f"artifact-integrity failure: missing columns {missing}")
            raise ValueError(f"artifact-integrity failure: missing columns {missing}")
        if len(cols) > 25:
            if has_objective:
                raise ChampionTrainingError("artifact-integrity failure: feature count exceeds 25")
            raise ValueError("artifact-integrity failure: feature count exceeds 25")
        select_cols = [*cols, "label_return", "decision_date"] + ([] if has_objective else [target_column])
        if has_objective:
            try:
                clean = train.select(select_cols).drop_nulls()
            except Exception as exc:
                raise ChampionTrainingError(f"artifact-integrity failure: column select failed: {exc}") from exc
        else:
            clean = train.select(select_cols).drop_nulls()
        # Exclude every non-finite feature/target row before fit.
        rows = clean.to_dicts()
        kept = []
        for r in rows:
            ok = True
            for c in cols:
                try:
                    if not math.isfinite(float(r[c])):
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            if not has_objective:
                try:
                    if not math.isfinite(float(r[target_column])):
                        ok = False
                except (AttributeError, IndexError, TypeError, ValueError):  # pragma: no cover - schema validation
                    ok = False
            try:
                if not math.isfinite(float(r["label_return"])):
                    ok = False
            except (AttributeError, IndexError, TypeError, ValueError):
                ok = False
            if ok:
                kept.append(r)
        if not kept:
            if has_objective:
                raise ChampionTrainingError("artifact-integrity failure: no finite training rows")
            raise ValueError("artifact-integrity failure: no finite training rows")
        # LambdaRank group sizes are positional; keep rows contiguous by date.
        kept.sort(key=lambda r: r["decision_date"])
        try:
            import lightgbm as lgb  # type: ignore[import]
            import numpy as np

            groups: dict[date, list[dict[str, object]]] = {}
            for r in kept:
                groups.setdefault(r["decision_date"], []).append(r)
            xs = np.ascontiguousarray([[float(r[c]) for c in cols] for r in kept], dtype=np.float64)
            params: dict[str, object] = {
                "verbosity": -1,
                "seed": self._seed,
                "deterministic": True,
                "force_col_wise": True,
                "num_threads": 1,
                "num_leaves": self._params["num_leaves"],
                "max_depth": self._params["max_depth"],
                "min_data_in_leaf": self._params["min_data_in_leaf"],
                "lambda_l2": 1.0,
                "feature_fraction": 1.0,
                "bagging_fraction": 1.0,
                "bagging_freq": 0,
            }
            if self._objective is not None:
                objective = self._objective
                ys = np.ascontiguousarray(objective.grade(clean.get_column('label_return').to_numpy()), dtype=np.int32)
                params["objective"] = "lambdarank"
                params["label_gain"] = [float(v) for v in objective.label_gain]
            else:
                ys = np.ascontiguousarray([float(r[target_column]) for r in kept], dtype=np.float64)
                params["objective"] = "regression"
            if xs.shape[0] != ys.shape[0] or xs.shape[0] == 0:
                raise ChampionTrainingError("artifact-integrity failure: empty feature/label matrix")
            grp = [len(groups[d]) for d in sorted(groups)]
            ds = lgb.Dataset(xs, label=ys, group=grp) if self._objective is not None else lgb.Dataset(xs, label=ys)
            model = lgb.train(params, ds, num_boost_round=50)
            try:
                probe = model.predict(xs[: min(1, xs.shape[0])])
                import math as _math

                for v in list(probe):
                    if not _math.isfinite(float(v)):
                        raise ChampionTrainingError("artifact-integrity failure: non-finite prediction")
            except ChampionTrainingError:
                raise
            except Exception as exc:
                raise ChampionTrainingError(f"artifact-integrity failure: prediction failed: {exc}") from exc
            ver = str(getattr(lgb, "__version__", "unknown"))
        except ChampionTrainingError:
            raise
        except Exception as exc:
            raise ChampionTrainingError(f"artifact-integrity failure: backend failed: {exc}") from exc
        feat_hash = hashlib.sha256(",".join(cols).encode()).hexdigest()[:16]
        panel_hash = hashlib.sha256(str(len(kept)).encode()).hexdigest()[:16]
        through = max(r["decision_date"] for r in kept)
        return ChampionModelArtifact(
            artifact_id=f"champion-{through.isoformat()}",
            trained_through=through,
            feature_columns=tuple(cols),
            model=model,
            feature_config_hash=feat_hash,
            model_config_hash=feat_hash,
            panel_hash=panel_hash,
            lgbm_version=ver,
            target_column=target_column,
            backend="lightgbm",
        )

    def score(self, snapshot: pl.DataFrame, *, artifact: ChampionModelArtifact) -> dict[str, float]:
        cols = list(artifact.feature_columns)
        missing = [c for c in cols if c not in snapshot.columns]
        if missing:
            raise ValueError(f"artifact-integrity failure: missing columns {missing}")
        if "source_ticker" not in snapshot.columns:
            raise ValueError("artifact-integrity failure: missing source_ticker")
        out: dict[str, float] = {}
        model = artifact.model
        for row in snapshot.iter_rows(named=True):
            ticker = str(row.get("source_ticker"))
            feats: list[float] = []
            finite = True
            for c in cols:
                try:
                    v = float(row[c])
                except Exception:
                    finite = False
                    break
                if not math.isfinite(v):
                    finite = False
                    break
                feats.append(v)
            if not finite:
                continue
            try:

                if hasattr(model, "predict"):
                    pred = float(model.predict([feats])[0])
                else:
                    pred = sum(feats) / len(feats) if feats else 0.0
            except (AttributeError, IndexError, TypeError, ValueError):
                # A corrupt/unusable artifact must not silently become a new
                # momentum model.  Callers can then preserve the position.
                continue
            out[ticker] = float(pred)
        return out


@dataclass(frozen=True)
class OosScoreStore:
    scores: pl.DataFrame

    def scores_for(self, decision_date: date, eligible_tickers: Collection[str]) -> dict[str, float]:
        frame = self.scores
        if frame.height == 0 or "decision_date" not in frame.columns:
            raise ValueError(f"OOS score missing for {decision_date}: empty store")
        day = frame.filter(pl.col("decision_date") == decision_date)
        if "is_evaluation" in frame.columns:
            day = day.filter(pl.col("is_evaluation") == True)  # noqa: E712
        if day.height == 0:
            raise ValueError(f"OOS score missing for {decision_date}: unscored date")
        allowed = set(eligible_tickers)
        out: dict[str, float] = {}
        for row in day.iter_rows(named=True):
            ticker = str(row.get("source_ticker"))
            if ticker not in allowed:
                continue
            try:
                out[ticker] = float(row.get("score"))
            except Exception:  # noqa: S112
                continue
        if not out:
            raise ValueError(f"OOS score missing for {decision_date}: no eligible tickers scored")
        return out

    def decision_for(self, decision_date: date, eligible_tickers: Collection[str]) -> tuple[dict[str, float], bool]:
        scores = self.scores_for(decision_date, eligible_tickers)
        frame = self.scores
        day = frame.filter(pl.col("decision_date") == decision_date)
        if "is_evaluation" in frame.columns:
            day = day.filter(pl.col("is_evaluation") == True)  # noqa: E712
        allowed = set(eligible_tickers)
        activate = False
        if "activate" in day.columns:
            for row in day.iter_rows(named=True):
                if str(row.get("source_ticker")) not in allowed:
                    continue
                if bool(row.get("activate")):
                    activate = True
                    break
        return scores, bool(activate)


def is_valid_champion_artifact(artifact: ChampionModelArtifact | object) -> bool:
    try:
        backend = getattr(artifact, "backend", None)
        if backend != "lightgbm":
            return False
        hurdle_models = getattr(artifact, "threshold_models", None)
        if hurdle_models is not None:
            return bool(hurdle_models) and bool(getattr(artifact, "artifact_hash", "")) and getattr(artifact, "trained_through", None) is not None
        ver = str(getattr(artifact, "lgbm_version", "") or "")
        if not ver or ver == "fallback-mean":
            return False
        for attr in ("feature_config_hash", "model_config_hash", "panel_hash"):
            val = str(getattr(artifact, attr, "") or "")
            if not val:
                return False
        through = getattr(artifact, "trained_through", None)
        if through is None:
            return False
        model = getattr(artifact, "model", None)
        if model is None:
            return False
        if isinstance(model, dict) and "mean" in model:
            return False
        return hasattr(model, "predict")
    except Exception:
        return False


__all__ = [
    "ChampionModelArtifact",
    "ChampionTailRanker",
    "ChampionTrainingError",
    "DateFold",
    "OosScoreStore",
    "PurgedDateWalkForward",
    "TailGradeObjective",
    "is_valid_champion_artifact",
]
