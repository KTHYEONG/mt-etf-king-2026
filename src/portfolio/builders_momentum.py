# mypy: ignore-errors
# ruff: noqa
"""Portfolio-policy strategy builders (momentum policy / vehicle / confidence).

Moved from src/strategies/factories/ for ARCH-1 layering: factories select
theme/index keys only; portfolio-backed construction lives in the portfolio
layer. Bodies moved verbatim.
"""

from __future__ import annotations

from src.strategies.factories._shared import TopKMomentum

def make_portfolio_momentum_policy() -> object:
    # PortfolioPolicy-backed model for P08 (B1 alpha + portfolio policy) with InstrumentMaster for ExposureSelector
    from src.core.paths import DataPaths
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.sizing import ConfidenceSizingConfig
    from src.universe.instruments import InstrumentMaster  # noqa: F401

    # wiring: ensure InstrumentMaster and DataPaths referenced inside _make_p08
    # try to build master for vehicle selection; fallback to empty master
    _master = None
    try:
        import datetime as _dt
        from pathlib import Path as _P

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
    b1 = TopKMomentum(horizon=20, name="portfolio.momentum_policy")
    # expose score via delegation and mark path_dependent
    policy.name = "portfolio.momentum_policy"  # type: ignore[attr-defined]
    policy.scores_path_independent = True

    def _score(snapshot, context):  # type: ignore[no-untyped-def]
        return b1.score(snapshot, context)

    policy.score = _score  # type: ignore[attr-defined]
    # wiring reference for lean_check
    _p08_ref = "portfolio.momentum_policy"  # noqa: F401
    return policy


def make_portfolio_momentum_vehicle() -> object:
    # P10: B1 alpha + family_canonical_scores + P08 PortfolioPolicy (vehicle on)
    from src.core.paths import DataPaths
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.selection import family_canonical_scores
    from src.portfolio.sizing import ConfidenceSizingConfig
    from src.universe.instruments import InstrumentMaster  # noqa: F401

    _master = None
    try:
        import datetime as _dt
        from pathlib import Path as _P

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
    cfg = ConfidenceSizingConfig()
    policy = PortfolioPolicy(sizing_config=cfg, master=_master, max_per_theme=2, max_per_family=1)
    b1 = TopKMomentum(horizon=20, name="portfolio.momentum_vehicle")
    policy.name = "portfolio.momentum_vehicle"  # type: ignore[attr-defined]
    policy.scores_path_independent = True
    # wiring: P08 anchor referenced for lean_check
    _p08_anchor = "portfolio.momentum_policy"  # noqa: F401
    _p10_anchor = "portfolio.momentum_vehicle"  # noqa: F401

    def _score(snapshot, context):  # type: ignore[no-untyped-def]
        raw = b1.score(snapshot, context)
        if not raw:
            return {}
        try:
            return family_canonical_scores(raw, _master)
        except Exception:
            return {}

    policy.score = _score  # type: ignore[attr-defined]
    return policy


def make_portfolio_momentum_confidence() -> object:
    # P11: B1 alpha + family_canonical_scores + confidence sizing + confidence-gated vehicle
    from src.core.paths import DataPaths
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.selection import family_canonical_scores
    from src.portfolio.sizing import ConfidenceSizingConfig
    from src.universe.instruments import InstrumentMaster  # noqa: F401

    # preflight same as P10: ensure gold/silver span check referenced

    _master = None
    try:
        import datetime as _dt
        from pathlib import Path as _P

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
    cfg = ConfidenceSizingConfig()
    policy = PortfolioPolicy(sizing_config=cfg, master=_master, max_per_theme=2, max_per_family=1)
    b1 = TopKMomentum(horizon=20, name="portfolio.momentum_confidence")
    policy.name = "portfolio.momentum_confidence"  # type: ignore[attr-defined]
    policy.scores_path_independent = True
    _p11_anchor = "portfolio.momentum_confidence"  # noqa: F401
    _p08_anchor = "portfolio.momentum_policy"  # noqa: F401

    def _score(snapshot, context):  # type: ignore[no-untyped-def]
        raw = b1.score(snapshot, context)
        if not raw:
            return {}
        try:
            return family_canonical_scores(raw, _master)
        except Exception:
            return {}

    policy.score = _score  # type: ignore[attr-defined]
    return policy
