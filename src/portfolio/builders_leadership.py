# mypy: ignore-errors
# ruff: noqa
"""Portfolio-policy strategy builders (leadership policy / confidence).

Moved from src/strategies/factories/ for ARCH-1 layering: factories select
theme/index keys only; portfolio-backed construction lives in the portfolio
layer. Bodies moved verbatim.
"""

from __future__ import annotations


def make_portfolio_leadership_policy() -> object:
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

    _master = None
    try:
        import datetime as _dt

        import polars as _pl

        from src.strategies.factories._shared import load_sponsor_context
        from src.universe.taxonomy import Taxonomy

        try:
            ctx = load_sponsor_context()
            _brand = ctx.brand_map
            _tax = ctx.taxonomy
        except Exception:
            _brand = {}
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
    policy.name = "portfolio.leadership_policy"  # type: ignore[attr-defined]
    policy.scores_path_independent = True

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
    _p12_anchor = "portfolio.leadership_policy"  # noqa: F401
    return policy


def make_portfolio_leadership_confidence() -> object:
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.intensity import FamilyIntensityConfig, FamilyIntensityModel
    from src.core.paths import DataPaths
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.selection import family_canonical_scores
    from src.portfolio.sizing import ConfidenceSizingConfig
    from src.universe.instruments import InstrumentMaster

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

        from src.strategies.factories._shared import load_sponsor_context
        from src.universe.taxonomy import Taxonomy

        try:
            ctx = load_sponsor_context()
            _brand = ctx.brand_map
            _tax = ctx.taxonomy
        except Exception:
            _brand = {}
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
    policy.name = "portfolio.leadership_confidence"  # type: ignore[attr-defined]
    policy.scores_path_independent = True

    def _score(snapshot, context):  # type: ignore[no-untyped-def]
        raw = family_model.score(snapshot, context)
        if not raw:
            return {}
        try:
            return family_canonical_scores(raw, _master)
        except Exception:
            return {}

    policy.score = _score  # type: ignore[attr-defined]
    _p13_anchor = "portfolio.leadership_confidence"  # noqa: F401
    return policy
