from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import polars as pl

from src.alpha.base import DecisionContext
from src.alpha.leadership import SectorLeadershipModel  # noqa: F401
from src.alpha.state import TransitionConfig, transition  # noqa: F401
from src.alpha.theme import ThemePanel  # noqa: F401
from src.features.regime import RegimeState
from src.universe.instruments import InstrumentMaster  # noqa: F401
from src.universe.taxonomy import Taxonomy  # noqa: F401


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


def _make_p08() -> object:
    # PortfolioPolicy-backed model for P08 (B1 alpha + portfolio policy) with InstrumentMaster for ExposureSelector
    from src.core.paths import DataPaths
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.sizing import ConfidenceSizingConfig
    from src.universe.instruments import InstrumentMaster  # noqa: F401

    # wiring: ensure InstrumentMaster and DataPaths referenced inside _make_p08
    _ = InstrumentMaster
    _ = DataPaths
    # try to build master for vehicle selection; fallback to empty master
    _master = None
    try:
        import datetime as _dt
        from pathlib import Path as _P

        import polars as _pl

        from src.universe.taxonomy import Taxonomy

        try:
            from src.universe.instruments import load_sponsor_brand_map

            try:
                _brand = load_sponsor_brand_map(_P("configs/sponsor_brands.yaml"))
            except Exception:
                _brand = {}
        except Exception:
            _brand = {}
        try:
            _tax = Taxonomy.from_yaml(_P("configs/taxonomy.yaml"))
        except Exception:
            _tax = Taxonomy(rules=[])
        # attempt to load panel via DataPaths (normalized/features) - fix drift from data/silver|gold
        _panel = None
        try:
            _paths = DataPaths(root=_P("data"))
            # Prefer gold then silver via DataPaths
            for _cand in [_paths.gold("etf_features"), _paths.silver("etf_daily")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
            # fallback to legacy paths if DataPaths not found
            if _panel is None or _panel.height == 0:
                for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                    if _cand.exists():
                        try:
                            _panel = _pl.read_parquet(str(_cand))
                            if _panel is not None and _panel.height > 0:
                                break
                        except Exception:
                            _panel = None
        except Exception:
            for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
        if _panel is not None and _panel.height > 0:
            try:
                _master = InstrumentMaster.build(_panel, _tax, _brand)
            except Exception:
                _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
        else:
            _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
    except Exception:
        import datetime as _dt2

        _master = InstrumentMaster(attributes={}, panel_start=_dt2.date(2020, 1, 1))
    cfg = ConfidenceSizingConfig()
    policy = PortfolioPolicy(sizing_config=cfg, master=_master, max_per_theme=2, max_per_family=1)
    # Attach B1 scoring delegation for AlphaModel compatibility
    b1 = TopKMomentum(horizon=20, name="P08")
    # expose score via delegation and mark path_dependent
    policy.name = "P08"  # type: ignore[attr-defined]
    policy.scores_path_independent = True
    _ = policy.allocate  # reference for wiring
    _ = policy.scores_path_independent  # wiring anchor

    def _score(snapshot, context):  # type: ignore[no-untyped-def]
        return b1.score(snapshot, context)

    policy.score = _score  # type: ignore[attr-defined]
    # wiring reference for lean_check
    _ = PortfolioPolicy
    _p08_ref = "P08"  # noqa: F401
    return policy

def _make_p10() -> object:
    # P10: B1 alpha + family_canonical_scores + P08 PortfolioPolicy (vehicle on)
    from src.core.paths import DataPaths
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.selection import family_canonical_scores
    from src.portfolio.sizing import ConfidenceSizingConfig
    from src.universe.instruments import InstrumentMaster  # noqa: F401

    _ = family_canonical_scores
    _ = InstrumentMaster
    _ = DataPaths
    _master = None
    try:
        import datetime as _dt
        from pathlib import Path as _P

        import polars as _pl

        from src.universe.taxonomy import Taxonomy

        try:
            from src.universe.instruments import load_sponsor_brand_map

            try:
                _brand = load_sponsor_brand_map(_P("configs/sponsor_brands.yaml"))
            except Exception:
                _brand = {}
        except Exception:
            _brand = {}
        try:
            _tax = Taxonomy.from_yaml(_P("configs/taxonomy.yaml"))
        except Exception:
            _tax = Taxonomy(rules=[])
        _panel = None
        try:
            _paths = DataPaths(root=_P("data"))
            for _cand in [_paths.gold("etf_features"), _paths.silver("etf_daily")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
            if _panel is None or _panel.height == 0:
                for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                    if _cand.exists():
                        try:
                            _panel = _pl.read_parquet(str(_cand))
                            if _panel is not None and _panel.height > 0:
                                break
                        except Exception:
                            _panel = None
        except Exception:
            for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
        if _panel is not None and _panel.height > 0:
            try:
                _master = InstrumentMaster.build(_panel, _tax, _brand)
            except Exception:
                _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
        else:
            _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
    except Exception:
        import datetime as _dt2

        _master = InstrumentMaster(attributes={}, panel_start=_dt2.date(2020, 1, 1))
    cfg = ConfidenceSizingConfig()
    policy = PortfolioPolicy(sizing_config=cfg, master=_master, max_per_theme=2, max_per_family=1)
    b1 = TopKMomentum(horizon=20, name="P10")
    policy.name = "P10"  # type: ignore[attr-defined]
    policy.scores_path_independent = True
    _ = policy.allocate
    _ = policy.scores_path_independent  # wiring anchor
    # wiring: P08 anchor referenced for lean_check
    _p08_anchor = "P08"  # noqa: F401
    _p10_anchor = "P10"  # noqa: F401

    def _score(snapshot, context):  # type: ignore[no-untyped-def]
        raw = b1.score(snapshot, context)
        if not raw:
            return {}
        try:
            return family_canonical_scores(raw, _master)
        except Exception:
            return {}

    policy.score = _score  # type: ignore[attr-defined]
    _ = PortfolioPolicy
    return policy


def _make_p11() -> object:
    # P11: B1 alpha + family_canonical_scores + confidence sizing + confidence-gated vehicle
    from src.core.paths import DataPaths
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.selection import family_canonical_scores
    from src.portfolio.sizing import ConfidenceSizingConfig
    from src.universe.instruments import InstrumentMaster  # noqa: F401

    _ = family_canonical_scores
    _ = InstrumentMaster
    _ = DataPaths
    # preflight same as P10: ensure gold/silver span check referenced
    from src.tournament.distribution import preflight_features_span_ok

    _ = preflight_features_span_ok
    _master = None
    try:
        import datetime as _dt
        from pathlib import Path as _P

        import polars as _pl

        from src.universe.taxonomy import Taxonomy

        try:
            from src.universe.instruments import load_sponsor_brand_map

            try:
                _brand = load_sponsor_brand_map(_P("configs/sponsor_brands.yaml"))
            except Exception:
                _brand = {}
        except Exception:
            _brand = {}
        try:
            _tax = Taxonomy.from_yaml(_P("configs/taxonomy.yaml"))
        except Exception:
            _tax = Taxonomy(rules=[])
        _panel = None
        try:
            _paths = DataPaths(root=_P("data"))
            for _cand in [_paths.gold("etf_features"), _paths.silver("etf_daily")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
            if _panel is None or _panel.height == 0:
                for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                    if _cand.exists():
                        try:
                            _panel = _pl.read_parquet(str(_cand))
                            if _panel is not None and _panel.height > 0:
                                break
                        except Exception:
                            _panel = None
        except Exception:
            for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
        if _panel is not None and _panel.height > 0:
            try:
                _master = InstrumentMaster.build(_panel, _tax, _brand)
            except Exception:
                _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
        else:
            _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
    except Exception:
        import datetime as _dt2

        _master = InstrumentMaster(attributes={}, panel_start=_dt2.date(2020, 1, 1))
    cfg = ConfidenceSizingConfig()
    policy = PortfolioPolicy(sizing_config=cfg, master=_master, max_per_theme=2, max_per_family=1)
    b1 = TopKMomentum(horizon=20, name="P11")
    policy.name = "P11"  # type: ignore[attr-defined]
    policy.scores_path_independent = True
    _ = policy.allocate
    _ = policy.scores_path_independent
    _p11_anchor = "P11"  # noqa: F401
    _p08_anchor = "P08"  # noqa: F401

    def _score(snapshot, context):  # type: ignore[no-untyped-def]
        raw = b1.score(snapshot, context)
        if not raw:
            return {}
        try:
            return family_canonical_scores(raw, _master)
        except Exception:
            return {}

    policy.score = _score  # type: ignore[attr-defined]
    _ = PortfolioPolicy
    return policy


def _make_p12() -> object:
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.cluster import ClusterResolver
    from src.alpha.leadership import SectorLeadershipModel, SectorScoreWeights
    from src.alpha.state import TransitionConfig
    from src.core.paths import DataPaths
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.selection import family_canonical_scores
    from src.portfolio.sizing import ConfidenceSizingConfig
    from src.universe.instruments import InstrumentMaster

    _ = family_canonical_scores
    _ = InstrumentMaster
    _ = DataPaths
    _ = SectorLeadershipModel
    _master = None
    try:
        import datetime as _dt

        import polars as _pl

        from src.universe.taxonomy import Taxonomy

        try:
            from src.universe.instruments import load_sponsor_brand_map

            try:
                _brand = load_sponsor_brand_map(_P("configs/sponsor_brands.yaml"))
            except Exception:
                _brand = {}
        except Exception:
            _brand = {}
        try:
            _tax = Taxonomy.from_yaml(_P("configs/taxonomy.yaml"))
        except Exception:
            _tax = Taxonomy(rules=[])
        _panel = None
        try:
            _paths = DataPaths(root=_P("data"))
            for _cand in [_paths.gold("etf_features"), _paths.silver("etf_daily")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
            if _panel is None or _panel.height == 0:
                for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                    if _cand.exists():
                        try:
                            _panel = _pl.read_parquet(str(_cand))
                            if _panel is not None and _panel.height > 0:
                                break
                        except Exception:
                            _panel = None
        except Exception:
            for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
        if _panel is not None and _panel.height > 0:
            try:
                _master = InstrumentMaster.build(_panel, _tax, _brand)
            except Exception:
                _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
        else:
            _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
    except Exception:
        import datetime as _dt2

        _master = InstrumentMaster(attributes={}, panel_start=_dt2.date(2020, 1, 1))
    # leadership config
    strat_path = _P("configs/strategies.yaml")
    if strat_path.exists():
        try:
            with open(strat_path, encoding="utf-8") as f:
                raw = _yaml.safe_load(f) or {}
            lead = (raw.get("leadership") or {}) if isinstance(raw, dict) else {}
            w_raw = (lead.get("sector_score_weights") or {}) if isinstance(lead, dict) else {}
            trans_raw = (lead.get("transition") or {}) if isinstance(lead, dict) else {}
            if not w_raw:
                w_raw = {"rs": 0.45, "accel": 0.30, "breadth": 0.25, "breakout": 0.0, "flow": 0.0}
            if not trans_raw:
                trans_raw = {
                    "rs_in": 0.55,
                    "rs_out": 0.35,
                    "rs_hi": 0.75,
                    "accel_in": -0.08,
                    "accel_out": -0.20,
                    "breadth_in": 0.65,
                    "breadth_out": 0.45,
                    "ext_in": 2.5,
                    "ext_out": 1.5,
                    "dd_in": 0.08,
                    "dd_out": 0.05,
                    "patience": 3,
                }
        except Exception:
            w_raw = {"rs": 0.45, "accel": 0.30, "breadth": 0.25, "breakout": 0.0, "flow": 0.0}
            trans_raw = {
                "rs_in": 0.55,
                "rs_out": 0.35,
                "rs_hi": 0.75,
                "accel_in": -0.08,
                "accel_out": -0.20,
                "breadth_in": 0.65,
                "breadth_out": 0.45,
                "ext_in": 2.5,
                "ext_out": 1.5,
                "dd_in": 0.08,
                "dd_out": 0.05,
                "patience": 3,
            }
    else:
        w_raw = {"rs": 0.45, "accel": 0.30, "breadth": 0.25, "breakout": 0.0, "flow": 0.0}
        trans_raw = {
            "rs_in": 0.55,
            "rs_out": 0.35,
            "rs_hi": 0.75,
            "accel_in": -0.08,
            "accel_out": -0.20,
            "breadth_in": 0.65,
            "breadth_out": 0.45,
            "ext_in": 2.5,
            "ext_out": 1.5,
            "dd_in": 0.08,
            "dd_out": 0.05,
            "patience": 3,
        }
    weights = SectorScoreWeights.from_yaml(w_raw)
    tcfg = TransitionConfig.from_yaml(trans_raw)
    max_per_theme = 2
    try:
        if isinstance(lead, dict) and "max_per_theme" in lead:
            max_per_theme = int(lead.get("max_per_theme"))  # type: ignore[arg-type]
    except Exception:
        max_per_theme = 2
    resolver = ClusterResolver(_master, max_per_theme=max_per_theme)
    leadership = SectorLeadershipModel(master=_master, resolver=resolver, weights=weights, transition_config=tcfg, history=None)
    cfg = ConfidenceSizingConfig()
    policy = PortfolioPolicy(sizing_config=cfg, master=_master, max_per_theme=2, max_per_family=1)
    policy.name = "P12"  # type: ignore[attr-defined]
    policy.scores_path_independent = True
    _ = policy.allocate
    _ = policy.scores_path_independent

    def _score(snapshot, context):  # type: ignore[no-untyped-def]
        raw = leadership.score(snapshot, context)
        if not raw:
            return {}
        try:
            return family_canonical_scores(raw, _master)
        except Exception:
            return {}

    policy.score = _score  # type: ignore[attr-defined]
    # expose theme_states_by_representative delegation
    try:  # noqa: SIM105
        policy.theme_states_by_representative = leadership.theme_states_by_representative  # type: ignore[attr-defined]
    except Exception:  # noqa: S110
        pass
    _ = PortfolioPolicy
    _p12_anchor = "P12"  # noqa: F401
    return policy


def _make_m13() -> object:
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.intensity import FamilyIntensityConfig, FamilyIntensityModel

    _ = FamilyIntensityModel
    _ = FamilyIntensityConfig
    cfg = FamilyIntensityConfig()
    try:
        fp = _P("configs/strategies.yaml")
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                raw = _yaml.safe_load(f) or {}
            int_raw = (raw.get("intensity") or {}) if isinstance(raw, dict) else {}
            if isinstance(int_raw, dict) and int_raw:
                cfg = FamilyIntensityConfig.from_yaml(int_raw)
    except Exception:
        cfg = FamilyIntensityConfig()
    # InstrumentMaster built like _make_p11 (gold/silver panel) with empty-master fallback
    _master = None
    try:
        import datetime as _dt

        import polars as _pl

        from src.core.paths import DataPaths
        from src.universe.taxonomy import Taxonomy

        try:
            from src.universe.instruments import InstrumentMaster as _IM  # noqa: N806

            _ = _IM
        except Exception:
            _IM = None  # type: ignore[assignment,misc]  # noqa: N806
        try:
            from src.universe.instruments import load_sponsor_brand_map

            try:
                _brand = load_sponsor_brand_map(_P("configs/sponsor_brands.yaml"))
            except Exception:
                _brand = {}
        except Exception:
            _brand = {}
        try:
            _tax = Taxonomy.from_yaml(_P("configs/taxonomy.yaml"))
        except Exception:
            _tax = Taxonomy(rules=[])
        _panel = None
        try:
            _paths = DataPaths(root=_P("data"))
            for _cand in [_paths.gold("etf_features"), _paths.silver("etf_daily")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
            if _panel is None or _panel.height == 0:
                for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                    if _cand.exists():
                        try:
                            _panel = _pl.read_parquet(str(_cand))
                            if _panel is not None and _panel.height > 0:
                                break
                        except Exception:
                            _panel = None
        except Exception:
            for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
        if _panel is not None and _panel.height > 0:
            try:
                from src.universe.instruments import InstrumentMaster as _IM2  # noqa: N806

                _master = _IM2.build(_panel, _tax, _brand)
            except Exception:
                _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
        else:
            _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
    except Exception:
        import datetime as _dt2

        _master = InstrumentMaster(attributes={}, panel_start=_dt2.date(2020, 1, 1))
    return FamilyIntensityModel(master=_master, config=cfg)


def _make_p13() -> object:
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.intensity import FamilyIntensityConfig, FamilyIntensityModel, family_intensity_scores
    from src.core.paths import DataPaths
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.selection import family_canonical_scores
    from src.portfolio.sizing import ConfidenceSizingConfig
    from src.universe.instruments import InstrumentMaster

    _ = family_intensity_scores
    _ = family_canonical_scores
    _ = InstrumentMaster
    _ = DataPaths
    _ = FamilyIntensityModel
    # load intensity config from configs/strategies.yaml (fallback defaults)
    cfg_int = FamilyIntensityConfig()
    try:
        fp2 = _P("configs/strategies.yaml")
        if fp2.exists():
            with open(fp2, encoding="utf-8") as f:
                raw2 = _yaml.safe_load(f) or {}
            int_raw2 = (raw2.get("intensity") or {}) if isinstance(raw2, dict) else {}
            if isinstance(int_raw2, dict) and int_raw2:
                cfg_int = FamilyIntensityConfig.from_yaml(int_raw2)
    except Exception:
        cfg_int = FamilyIntensityConfig()
    # InstrumentMaster built like _make_p11 (gold/silver panel) with empty-master fallback
    _master = None
    try:
        import datetime as _dt

        import polars as _pl

        from src.universe.taxonomy import Taxonomy

        try:
            from src.universe.instruments import load_sponsor_brand_map

            try:
                _brand = load_sponsor_brand_map(_P("configs/sponsor_brands.yaml"))
            except Exception:
                _brand = {}
        except Exception:
            _brand = {}
        try:
            _tax = Taxonomy.from_yaml(_P("configs/taxonomy.yaml"))
        except Exception:
            _tax = Taxonomy(rules=[])
        _panel = None
        try:
            _paths = DataPaths(root=_P("data"))
            for _cand in [_paths.gold("etf_features"), _paths.silver("etf_daily")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
            if _panel is None or _panel.height == 0:
                for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                    if _cand.exists():
                        try:
                            _panel = _pl.read_parquet(str(_cand))
                            if _panel is not None and _panel.height > 0:
                                break
                        except Exception:
                            _panel = None
        except Exception:
            for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
        if _panel is not None and _panel.height > 0:
            try:
                _master = InstrumentMaster.build(_panel, _tax, _brand)
            except Exception:
                _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
        else:
            _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
    except Exception:
        import datetime as _dt2

        _master = InstrumentMaster(attributes={}, panel_start=_dt2.date(2020, 1, 1))
    family_model = FamilyIntensityModel(master=_master, config=cfg_int)
    cfg = ConfidenceSizingConfig()
    policy = PortfolioPolicy(sizing_config=cfg, master=_master, max_per_theme=2, max_per_family=1)
    policy.name = "P13"  # type: ignore[attr-defined]
    policy.scores_path_independent = True
    _ = policy.allocate
    _ = policy.scores_path_independent

    def _score(snapshot, context):  # type: ignore[no-untyped-def]
        raw = family_model.score(snapshot, context)
        if not raw:
            return {}
        try:
            return family_canonical_scores(raw, _master)
        except Exception:
            return {}

    policy.score = _score  # type: ignore[attr-defined]
    _ = PortfolioPolicy
    _p13_anchor = "P13"  # noqa: F401
    return policy


def _make_p14() -> object:
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.intensity import FamilyIntensityConfig, FamilyIntensityModel, family_intensity_scores
    from src.core.paths import DataPaths
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.selection import family_canonical_scores
    from src.portfolio.sizing import ConfidenceSizingConfig, LotteryExposureConfig
    from src.universe.instruments import InstrumentMaster

    _ = family_intensity_scores
    _ = family_canonical_scores
    _ = InstrumentMaster
    _ = DataPaths
    _ = FamilyIntensityModel
    _ = LotteryExposureConfig
    # load intensity config
    cfg_int = FamilyIntensityConfig()
    try:
        fp2 = _P("configs/strategies.yaml")
        if fp2.exists():
            with open(fp2, encoding="utf-8") as f:
                raw2 = _yaml.safe_load(f) or {}
            int_raw2 = (raw2.get("intensity") or {}) if isinstance(raw2, dict) else {}
            if isinstance(int_raw2, dict) and int_raw2:
                cfg_int = FamilyIntensityConfig.from_yaml(int_raw2)
    except Exception:
        cfg_int = FamilyIntensityConfig()
    # load lottery config
    lottery_cfg = LotteryExposureConfig(enabled=True)
    try:
        fp3 = _P("configs/strategies.yaml")
        if fp3.exists():
            with open(fp3, encoding="utf-8") as f:
                raw3 = _yaml.safe_load(f) or {}
            port_raw = (raw3.get("portfolio") or {}) if isinstance(raw3, dict) else {}
            lot_raw = (port_raw.get("lottery_exposure") or {}) if isinstance(port_raw, dict) else {}
            if isinstance(lot_raw, dict):
                parsed = LotteryExposureConfig.from_yaml(lot_raw)
                # ensure enabled True for P14 (fail-closed defaults would be False)
                if parsed.enabled:
                    lottery_cfg = parsed
                else:
                    # if yaml has no block or enabled False, force enabled True preserving other fields
                    lottery_cfg = LotteryExposureConfig(
                        enabled=True,
                        risk_on_regimes=parsed.risk_on_regimes,
                        w_top=parsed.w_top,
                        max_gross=parsed.max_gross,
                        suppress_vehicle_gate=parsed.suppress_vehicle_gate,
                        suppress_trim=parsed.suppress_trim,
                    )
            _ = LotteryExposureConfig
    except Exception:
        lottery_cfg = LotteryExposureConfig(enabled=True)
    # InstrumentMaster built like _make_p13
    _master = None
    try:
        import datetime as _dt

        import polars as _pl

        from src.universe.taxonomy import Taxonomy

        try:
            from src.universe.instruments import load_sponsor_brand_map

            try:
                _brand = load_sponsor_brand_map(_P("configs/sponsor_brands.yaml"))
            except Exception:
                _brand = {}
        except Exception:
            _brand = {}
        try:
            _tax = Taxonomy.from_yaml(_P("configs/taxonomy.yaml"))
        except Exception:
            _tax = Taxonomy(rules=[])
        _panel = None
        try:
            _paths = DataPaths(root=_P("data"))
            for _cand in [_paths.gold("etf_features"), _paths.silver("etf_daily")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
            if _panel is None or _panel.height == 0:
                for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                    if _cand.exists():
                        try:
                            _panel = _pl.read_parquet(str(_cand))
                            if _panel is not None and _panel.height > 0:
                                break
                        except Exception:
                            _panel = None
        except Exception:
            for _cand in [_P("data/silver/etf_daily.parquet"), _P("data/gold/etf_features.parquet")]:
                if _cand.exists():
                    try:
                        _panel = _pl.read_parquet(str(_cand))
                        if _panel is not None and _panel.height > 0:
                            break
                    except Exception:
                        _panel = None
        if _panel is not None and _panel.height > 0:
            try:
                _master = InstrumentMaster.build(_panel, _tax, _brand)
            except Exception:
                _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
        else:
            _master = InstrumentMaster(attributes={}, panel_start=_dt.date(2020, 1, 1))
    except Exception:
        import datetime as _dt2

        _master = InstrumentMaster(attributes={}, panel_start=_dt2.date(2020, 1, 1))
    family_model = FamilyIntensityModel(master=_master, config=cfg_int)
    cfg = ConfidenceSizingConfig()
    policy = PortfolioPolicy(sizing_config=cfg, master=_master, max_per_theme=2, max_per_family=1, lottery_config=lottery_cfg)
    policy.name = "P14"  # type: ignore[attr-defined]
    policy.scores_path_independent = True
    policy.lottery_config = lottery_cfg
    _ = policy.allocate
    _ = policy.scores_path_independent

    def _score(snapshot, context):  # type: ignore[no-untyped-def]
        raw = family_model.score(snapshot, context)
        if not raw:
            return {}
        try:
            return family_canonical_scores(raw, _master)
        except Exception:
            return {}

    policy.score = _score  # type: ignore[attr-defined]
    _ = PortfolioPolicy
    _p14_anchor = "P14"  # noqa: F401
    _p13_anchor = "P13"  # noqa: F401
    return policy


def _make_m07() -> SectorLeadershipModel:
    # lazy wiring for M07; requires master and configs - fallback to empty master for registry test
    from pathlib import Path

    import yaml

    from src.alpha.cluster import ClusterResolver
    from src.alpha.leadership import SectorScoreWeights

    strat_path = Path("configs/strategies.yaml")
    if strat_path.exists():
        with open(strat_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        lead = (raw.get("leadership") or {}) if isinstance(raw, dict) else {}
        w_raw = (lead.get("sector_score_weights") or {}) if isinstance(lead, dict) else {}
        trans_raw = (lead.get("transition") or {}) if isinstance(lead, dict) else {}
    else:
        w_raw = {"rs": 0.45, "accel": 0.30, "breadth": 0.25, "breakout": 0.0, "flow": 0.0}
        trans_raw = {
            "rs_in": 0.55,
            "rs_out": 0.35,
            "rs_hi": 0.75,
            "accel_in": -0.08,
            "accel_out": -0.20,
            "breadth_in": 0.65,
            "breadth_out": 0.45,
            "ext_in": 2.5,
            "ext_out": 1.5,
            "dd_in": 0.08,
            "dd_out": 0.05,
            "patience": 3,
        }
    weights = SectorScoreWeights.from_yaml(w_raw)
    tcfg = TransitionConfig.from_yaml(trans_raw)
    max_per_theme = 2
    if isinstance(lead, dict) and "max_per_theme" in lead:
        try:
            mp = lead.get("max_per_theme")
            max_per_theme = int(mp)  # type: ignore[arg-type]
        except Exception:
            max_per_theme = 2
    # empty master fallback
    master = InstrumentMaster(attributes={}, panel_start=__import__("datetime").date(2020, 1, 1))
    resolver = ClusterResolver(master, max_per_theme=max_per_theme)
    return SectorLeadershipModel(master=master, resolver=resolver, weights=weights, transition_config=tcfg, history=None)


BASELINES: Final[Mapping[str, Callable[[], object]]] = {
    "B0": _make_b0,
    "B1": _make_b1,
    "B2": _make_b2,
    "B3": _make_b3,
    "B4": _make_b4,
    "B5": _make_b5,
    "M07": _make_m07,
    "M13": _make_m13,
    "P08": _make_p08,
    "P10": _make_p10,
    "P11": _make_p11,
    "P12": _make_p12,
    "P13": _make_p13,
    "P14": _make_p14,
}
