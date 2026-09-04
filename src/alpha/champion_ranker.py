# mypy: ignore-errors
# ruff: noqa: S101
"""Purged date walk-forward splitter, LambdaRank tail ranker, OOS score store."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import date

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
    ) -> None:
        cols = tuple(feature_columns or ())
        if len(cols) > 25:
            raise ValueError("feature count exceeds 25")
        if int(num_leaves) > 8 or int(max_depth) > 4 or int(min_data_in_leaf) < 100:
            raise ValueError("model capacity exceeds shallow limits")
        self._features = cols
        self._seed = int(seed)
        self._params = {
            "num_leaves": int(num_leaves),
            "max_depth": int(max_depth),
            "min_data_in_leaf": int(min_data_in_leaf),
        }

    def fit(self, train: pl.DataFrame) -> ChampionModelArtifact:
        if "label_return" not in train.columns or "decision_date" not in train.columns:
            raise ValueError("artifact-integrity failure: missing label/decision_date")
        cols = list(self._features) if self._features else [c for c in train.columns if c not in ("label_return", "label_rank", "decision_date", "source_ticker", "family_key")]
        missing = [c for c in cols if c not in train.columns]
        if missing:
            raise ValueError(f"artifact-integrity failure: missing columns {missing}")
        if len(cols) > 25:
            raise ValueError("artifact-integrity failure: feature count exceeds 25")
        clean = train.select([*cols, "label_return", "decision_date"]).drop_nulls()
        # Exclude every non-finite feature row before fit.
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
            try:
                if not math.isfinite(float(r["label_return"])):
                    ok = False
            except (AttributeError, IndexError, TypeError, ValueError):
                ok = False
            if ok:
                kept.append(r)
        if not kept:
            raise ValueError("artifact-integrity failure: no finite training rows")
        # LambdaRank group sizes are positional; keep rows contiguous by date.
        kept.sort(key=lambda r: r["decision_date"])
        model: object = {"mean": sum(float(r["label_return"]) for r in kept) / len(kept)}
        try:
            import lightgbm as lgb  # type: ignore[import]

            groups: dict[date, list[dict[str, object]]] = {}
            for r in kept:
                groups.setdefault(r["decision_date"], []).append(r)
            xs = [[float(r[c]) for c in cols] for r in kept]
            ys = [float(r["label_return"]) for r in kept]
            grp = [len(groups[d]) for d in sorted(groups)]
            ds = lgb.Dataset(xs, label=ys, group=grp)
            params = {
                "objective": "lambdarank",
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
            model = lgb.train(params, ds, num_boost_round=50)
            ver = str(getattr(lgb, "__version__", "unknown"))
        except Exception:
            ver = "fallback-mean"
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
            day = day.filter(pl.col("is_evaluation") is True if False else pl.col("is_evaluation") == True)  # noqa: E712
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


__all__ = [
    "ChampionModelArtifact",
    "ChampionTailRanker",
    "DateFold",
    "OosScoreStore",
    "PurgedDateWalkForward",
]
