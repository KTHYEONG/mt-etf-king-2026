# mypy: ignore-errors
# ruff: noqa
"""Split-fill sticky model (moved from src.portfolio.split_fill for ARCH-1 layering).

The model composes sticky-leader scoring with the split-fill residual allocator;
it lives with the sticky strategies so factories need not import the portfolio layer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import polars as pl

from src.portfolio.split_fill import split_residual_plus2

def families_from_snapshot(snapshot: object) -> dict[str, str]:
    if snapshot is None or not isinstance(snapshot, pl.DataFrame):
        return {}
    try:
        if snapshot.height == 0 or snapshot.width == 0:
            return {}
    except Exception:
        return {}
    if "ticker" not in snapshot.columns:
        return {}
    out: dict[str, str] = {}
    has_underlying = "underlying_index_name" in snapshot.columns
    try:
        for row in snapshot.iter_rows(named=True):
            try:
                t = str(row.get("ticker"))
            except Exception:
                continue
            if not t:
                continue
            family = t
            if has_underlying:
                try:
                    val = row.get("underlying_index_name")
                    if val is not None:
                        s = str(val).strip()
                        if s and s.lower() not in ("none", "nan", ""):
                            try:
                                if isinstance(val, float) and not math.isfinite(val):
                                    family = t
                                else:
                                    family = s
                            except Exception:
                                family = s
                        else:
                            family = t
                    else:
                        family = t
                except Exception:
                    family = t
            out[t] = str(family)
    except Exception:
        return out
    return out


class SplitFillStickyModel:
    name: str
    config: object
    _inner: object
    _families: dict[str, str]

    def __init__(self, name: str = "sticky.split_fill_lock", config: object | None = None) -> None:
        from src.alpha.sticky import StickyLeaderConfig, StickyLeaderModel

        self.name = str(name)
        if config is not None:
            self.config = config
        else:
            self.config = StickyLeaderConfig(
                min_gap=0.08,
                min_hold=3,
                impulse_gap=0.04,
                cash_drawdown=-0.12,
                collapse_family=True,
                lock_level=0.40,
            )
            # ensure defaults from StickyLeaderConfig match
            try:
                # ensure lock_level set correctly even if from_yaml defaults differ
                self.config.lock_level = 0.40  # type: ignore[attr-defined]
            except Exception:
                pass
        # inner model
        try:
            self._inner = StickyLeaderModel(name=self.name, config=self.config)  # type: ignore[arg-type]
        except Exception:
            from src.alpha.sticky import StickyLeaderModel as _SLM

            self._inner = _SLM(name=self.name, config=self.config)  # type: ignore[arg-type]
        # alias for test compatibility
        self.inner = self._inner
        self._families: dict[str, str] = {}

    def score(self, snapshot: pl.DataFrame, context: object) -> dict[str, float]:
        # store families
        try:
            self._families = families_from_snapshot(snapshot)
        except Exception:
            self._families = {}
        # delegate
        try:
            fn = getattr(self._inner, "score", None)
            if callable(fn):
                return fn(snapshot, context)  # type: ignore[no-any-return]
        except Exception:
            pass
        return {}

    def allocate(
        self,
        scores: Mapping[str, float],
        adv: Mapping[str, float] | None = None,
        participation: float | None = None,
        capital: float | None = None,
        current_weights: Mapping[str, float] | None = None,
        **kwargs: object,
    ) -> object:
        # handle aliases: adv_by_ticker, equity, etc.
        if adv is None and "adv_by_ticker" in kwargs:
            try:
                adv = kwargs.get("adv_by_ticker")  # type: ignore[assignment]
            except Exception:
                adv = None
        if participation is None and "participation" in kwargs:
            try:
                v = kwargs.get("participation")
                if v is not None:
                    participation = float(v)  # type: ignore[arg-type]
            except Exception:
                pass
        if participation is None:
            participation = 0.01
        # capital/equity
        eq: float | None = None
        if capital is not None:
            eq = float(capital)  # type: ignore[arg-type]
        elif "equity" in kwargs:
            try:
                eq = float(kwargs.get("equity"))  # type: ignore[arg-type]
            except Exception:
                eq = None
        elif "capital" in kwargs:
            try:
                eq = float(kwargs.get("capital"))  # type: ignore[arg-type]
            except Exception:
                eq = None
        if eq is None:
            eq = 1_000_000_000.0
        if adv is None:
            adv = {}
        if current_weights is None and "current_weights" in kwargs:
            try:
                current_weights = kwargs.get("current_weights")  # type: ignore[assignment]
            except Exception:
                current_weights = {}
        if current_weights is None:
            current_weights = {}
        # handle empty scores
        if not scores:
            try:
                from src.portfolio.policy import PortfolioDecision

                return PortfolioDecision(weights={}, rationale={}, vehicles={}, gross=0.0)
            except Exception:
                return {}
        # TOP1 leader
        try:
            from src.portfolio.sizing import SizingScheme, weights_from_scores

            top = weights_from_scores(scores, SizingScheme.TOP1, k=1)
            if top:
                leader = next(iter(top.keys()))
            else:
                # fallback to max score
                leader = sorted(scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))[0][0]
        except Exception:
            try:
                leader = sorted(scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))[0][0]
            except Exception:
                try:
                    from src.portfolio.policy import PortfolioDecision

                    return PortfolioDecision(weights={}, rationale={}, vehicles={}, gross=0.0)
                except Exception:
                    return {}
        # family mapping: use stored else identity
        fam = self._families if self._families else {str(k): str(k) for k in scores.keys()}
        # ensure leader in fam? if not, add
        if str(leader) not in fam:
            fam[str(leader)] = str(leader)
        # call split
        res = split_residual_plus2(str(leader), scores, adv, family_by_ticker=fam, equity=float(eq), participation=float(participation))
        weights = dict(res.weights) if res.weights else {}
        # return PortfolioDecision for engine compatibility
        try:
            from src.portfolio.policy import PortfolioDecision

            return PortfolioDecision(weights=weights, rationale={}, vehicles={}, gross=None)
        except Exception:
            return weights
