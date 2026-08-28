from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import polars as pl

from src.alpha.base import DecisionContext
from src.features.regime import RegimeState


class BuyAndHoldBaseline:
    def __init__(self, ticker: str, name: str = "B0") -> None:
        self.ticker = ticker
        self.name = name

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]:
        # Hold single ticker if present in snapshot
        if snapshot.height > 0 and "ticker" in snapshot.columns:
            tickers = snapshot.select(pl.col("ticker")).to_series().to_list()
            if self.ticker in tickers:
                return {self.ticker: 1.0}
        return {self.ticker: 1.0} if self.ticker else {}


class TopKMomentum:
    def __init__(self, horizon: int = 20, name: str = "B1") -> None:
        self.horizon = horizon
        self.name = name

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]:
        if snapshot.height == 0:
            return {}
        col = f"mom_{self.horizon}"
        # Prefer column if exists, else try mom_20 or first mom_*
        if col not in snapshot.columns:
            # fallback to any mom_ column
            cand = [c for c in snapshot.columns if c.startswith("mom_") and not c.endswith("_rs") and not c.endswith("_z")]
            if not cand:
                return {}
            col = cand[0]
        scores: dict[str, float] = {}
        # Need ticker and col
        if "ticker" not in snapshot.columns:
            return {}
        for row in snapshot.iter_rows(named=True):
            t = str(row.get("ticker"))
            v = row.get(col)
            if v is None:
                continue
            try:
                scores[t] = float(v)
            except Exception:
                continue
        return scores


class MomentumTrendFilter:
    def __init__(self, horizon: int = 20, ma_window: int = 20, name: str = "B3") -> None:
        self.horizon = horizon
        self.ma_window = ma_window
        self.name = name

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]:
        if snapshot.height == 0:
            return {}
        mom_col = f"mom_{self.horizon}"
        if mom_col not in snapshot.columns:
            cand = [c for c in snapshot.columns if c.startswith("mom_")]
            if cand:
                mom_col = cand[0]
            else:
                return {}
        ma_col = f"ma_{self.ma_window}"
        # If ma column not present, try trend column or use mom filter without?
        scores: dict[str, float] = {}
        for row in snapshot.iter_rows(named=True):
            t = str(row.get("ticker"))
            v = row.get(mom_col)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            # Trend filter: close > MA20 . Need ma column or close/ma comparison
            # Check if ma column exists, else look for 'ma_20' or 'trend' bool
            passed = True
            if ma_col in snapshot.columns:
                mv = row.get(ma_col)
                close = row.get("close")
                if mv is not None and close is not None:
                    try:
                        passed = float(close) > float(mv)
                    except Exception:
                        passed = True
                else:
                    passed = False
            elif "close" in row and ma_col in row:
                # handled above
                pass
            else:
                # fallback: if 'trend' column exists?
                if "trend" in row:
                    passed = bool(row.get("trend"))
            if passed:
                scores[t] = fv
        return scores


class ThemeMomentum:
    def __init__(self, horizon: int = 20, name: str = "B4") -> None:
        self.horizon = horizon
        self.name = name

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]:
        if snapshot.height == 0:
            return {}
        # Group by theme, compute theme momentum as mean of mom_horizon within theme, then select best theme's members
        mom_col = f"mom_{self.horizon}"
        if mom_col not in snapshot.columns:
            cand = [c for c in snapshot.columns if c.startswith("mom_")]
            if cand:
                mom_col = cand[0]
            else:
                return {}
        # Determine theme column: try 'theme' then 'underlying_index_name'
        theme_col = None
        for c in ["theme", "underlying_index_name", "idx_ind_nm", "index_key"]:
            if c in snapshot.columns:
                theme_col = c
                break
        if theme_col is None:
            # fallback to per-ticker scoring like TopKMomentum
            scores: dict[str, float] = {}
            for row in snapshot.iter_rows(named=True):
                t = str(row.get("ticker"))
                v = row.get(mom_col)
                if v is None:
                    continue
                try:
                    scores[t] = float(v)
                except Exception:
                    continue
            return scores
        # Compute theme average
        # Use polars group_by
        try:
            grp = snapshot.filter(pl.col(mom_col).is_not_null()).group_by(theme_col).agg(pl.col(mom_col).mean().alias("theme_mom"))
            if grp.height == 0:
                return {}
            # Find theme with max avg
            max_row = grp.sort("theme_mom", descending=True).head(1)
            best_theme = max_row.select(pl.col(theme_col)).to_series().to_list()[0]
            # Return scores for tickers belonging to best theme
            filtered = snapshot.filter(pl.col(theme_col) == best_theme)
            scores2: dict[str, float] = {}
            for row in filtered.iter_rows(named=True):
                t = str(row.get("ticker"))
                v = row.get(mom_col)
                if v is None:
                    continue
                try:
                    scores2[t] = float(v)
                except Exception:
                    continue
            return scores2
        except Exception:
            # fallback
            scores3: dict[str, float] = {}
            for row in snapshot.iter_rows(named=True):
                t = str(row.get("ticker"))
                v = row.get(mom_col)
                if v is None:
                    continue
                try:
                    scores3[t] = float(v)
                except Exception:
                    continue
            return scores3


class RegimeGatedMomentum:
    def __init__(self, inner: object, blocked: frozenset[RegimeState], name: str = "B5") -> None:
        self.inner = inner
        self.blocked = blocked
        self.name = name

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]:
        # If regime in blocked set, return empty (full cash)
        regime = context.regime
        if regime is not None:
            # regime may be RegimeSnapshot with .state
            state = getattr(regime, "state", None)
            if state in self.blocked:
                return {}
        # Delegate
        inner_score = getattr(self.inner, "score", None)
        if inner_score is not None:
            return inner_score(snapshot, context)  # type: ignore[no-any-return]
        return {}


def _make_b0() -> BuyAndHoldBaseline:
    # Default ticker for buy-and-hold: KODEX 200 = 069500
    return BuyAndHoldBaseline(ticker="069500", name="B0")


def _make_b1() -> TopKMomentum:
    return TopKMomentum(horizon=20, name="B1")


def _make_b2() -> TopKMomentum:
    # B2 is Top-3 equal weighted; same scoring as B1 but sizing differs via config. For baseline registry, we provide same model with name B2
    return TopKMomentum(horizon=20, name="B2")


def _make_b3() -> MomentumTrendFilter:
    return MomentumTrendFilter(horizon=20, ma_window=20, name="B3")


def _make_b4() -> ThemeMomentum:
    return ThemeMomentum(horizon=20, name="B4")


def _make_b5() -> RegimeGatedMomentum:
    inner = ThemeMomentum(horizon=20, name="B5_inner")
    blocked = frozenset({RegimeState.STRONG_RISK_OFF})
    return RegimeGatedMomentum(inner=inner, blocked=blocked, name="B5")


BASELINES: Final[Mapping[str, Callable[[], object]]] = {
    "B0": _make_b0,
    "B1": _make_b1,
    "B2": _make_b2,
    "B3": _make_b3,
    "B4": _make_b4,
    "B5": _make_b5,
}
