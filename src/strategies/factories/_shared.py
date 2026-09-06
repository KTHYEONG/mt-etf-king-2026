# mypy: ignore-errors
# ruff: noqa
"""Shared baseline classes and sponsor/taxonomy context cache (P3 split)."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import polars as pl

from src.alpha.base import DecisionContext
from src.alpha.leadership import SectorLeadershipModel  # noqa: F401
from src.alpha.state import TransitionConfig, transition  # noqa: F401
from src.alpha.theme import ThemePanel  # noqa: F401
from src.features.regime import RegimeState
from src.strategies.baselines.core import BuyAndHoldBaseline  # noqa: F401
from src.universe.instruments import InstrumentMaster  # noqa: F401


@dataclass(frozen=True)
class SponsorContext:
    brand_map: object
    taxonomy: object


@functools.lru_cache(maxsize=1)
def load_sponsor_context() -> SponsorContext:
    """Single cached load of the sponsor brand map + taxonomy (P1: was reloaded 4x in baselines.py)."""
    from src.core.config import config_path
    from src.universe.instruments import load_sponsor_brand_map
    from src.universe.taxonomy import Taxonomy

    brand_map = load_sponsor_brand_map(config_path("sponsor_brands"))
    taxonomy = Taxonomy.from_yaml(config_path("taxonomy"))
    return SponsorContext(brand_map=brand_map, taxonomy=taxonomy)


class TopKMomentum:
    def __init__(self, horizon: int = 20, name: str = "baseline.mom20_top1") -> None:
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
    def __init__(self, horizon: int = 20, ma_window: int = 20, name: str = "baseline.mom20_ma_gate") -> None:
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
    def __init__(self, horizon: int = 20, name: str = "baseline.theme_momentum") -> None:
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
    def __init__(self, inner: object, blocked: frozenset[RegimeState], name: str = "baseline.regime_gated_theme") -> None:
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

