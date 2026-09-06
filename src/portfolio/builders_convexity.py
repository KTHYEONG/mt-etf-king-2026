# mypy: ignore-errors
# ruff: noqa
"""Portfolio-policy strategy builders (lottery / convexity).

Moved from src/strategies/factories/ for ARCH-1 layering: factories select
theme/index keys only; portfolio-backed construction lives in the portfolio
layer. Bodies moved verbatim.
"""

from __future__ import annotations


def make_portfolio_lottery_exposure() -> object:
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.intensity import FamilyIntensityConfig, FamilyIntensityModel
    from src.core.paths import DataPaths
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.selection import family_canonical_scores
    from src.portfolio.sizing import ConfidenceSizingConfig, LotteryExposureConfig
    from src.universe.instruments import InstrumentMaster

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
    except Exception:
        lottery_cfg = LotteryExposureConfig(enabled=True)
    # InstrumentMaster built like _make_p13
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
    policy = PortfolioPolicy(sizing_config=cfg, master=_master, max_per_theme=2, max_per_family=1, lottery_config=lottery_cfg)
    policy.name = "portfolio.lottery_exposure"  # type: ignore[attr-defined]
    policy.scores_path_independent = True
    policy.lottery_config = lottery_cfg

    def _score(snapshot, context):  # type: ignore[no-untyped-def]
        raw = family_model.score(snapshot, context)
        if not raw:
            return {}
        try:
            return family_canonical_scores(raw, _master)
        except Exception:
            return {}

    policy.score = _score  # type: ignore[attr-defined]
    _p14_anchor = "portfolio.lottery_exposure"  # noqa: F401
    _p13_anchor = "portfolio.leadership_confidence"  # noqa: F401
    return policy


def make_portfolio_tail_concentration() -> object:
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.intensity import FamilyIntensityConfig, FamilyIntensityModel
    from src.core.paths import DataPaths
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.selection import family_canonical_scores
    from src.portfolio.sizing import ConfidenceSizingConfig, LotteryExposureConfig
    from src.universe.instruments import InstrumentMaster

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
    # load lottery config with selective capacity route enabled: suppress_vehicle_gate false, suppress_trim false
    lottery_cfg = LotteryExposureConfig(enabled=True, suppress_vehicle_gate=False, suppress_trim=False)
    try:
        fp3 = _P("configs/strategies.yaml")
        if fp3.exists():
            with open(fp3, encoding="utf-8") as f:
                raw3 = _yaml.safe_load(f) or {}
            port_raw = (raw3.get("portfolio") or {}) if isinstance(raw3, dict) else {}
            lot_raw = (port_raw.get("lottery_exposure") or {}) if isinstance(port_raw, dict) else {}
            if isinstance(lot_raw, dict) and lot_raw:
                parsed = LotteryExposureConfig.from_yaml(lot_raw)
                # enforce selective capacity route: suppress_vehicle_gate false, suppress_trim false, enabled true
                lottery_cfg = LotteryExposureConfig(
                    enabled=True,
                    risk_on_regimes=parsed.risk_on_regimes,
                    w_top=parsed.w_top,
                    max_gross=parsed.max_gross,
                    suppress_vehicle_gate=False,
                    suppress_trim=False,
                )
    except Exception:
        lottery_cfg = LotteryExposureConfig(enabled=True, suppress_vehicle_gate=False, suppress_trim=False)
    # InstrumentMaster
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
    policy = PortfolioPolicy(sizing_config=cfg, master=_master, max_per_theme=2, max_per_family=1, lottery_config=lottery_cfg)
    policy.name = "portfolio.tail_concentration"  # type: ignore[attr-defined]
    policy.scores_path_independent = True
    policy.lottery_config = lottery_cfg

    def _score(snapshot, context):  # type: ignore[no-untyped-def]
        raw = family_model.score(snapshot, context)
        if not raw:
            return {}
        try:
            return family_canonical_scores(raw, _master)
        except Exception:
            return {}

    policy.score = _score  # type: ignore[attr-defined]
    _p15_anchor = "portfolio.tail_concentration"  # noqa: F401
    return policy


def make_portfolio_convexity_hold() -> object:
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.intensity import FamilyIntensityConfig, FamilyIntensityModel
    from src.core.paths import DataPaths
    from src.portfolio.convexity import ConvexityHoldConfig
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.selection import family_canonical_scores
    from src.portfolio.sizing import ConfidenceSizingConfig, LotteryExposureConfig
    from src.universe.instruments import InstrumentMaster

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
                if parsed.enabled:
                    lottery_cfg = parsed
                else:
                    lottery_cfg = LotteryExposureConfig(
                        enabled=True,
                        risk_on_regimes=parsed.risk_on_regimes,
                        w_top=parsed.w_top,
                        max_gross=parsed.max_gross,
                        suppress_vehicle_gate=parsed.suppress_vehicle_gate,
                        suppress_trim=parsed.suppress_trim,
                    )
    except Exception:
        lottery_cfg = LotteryExposureConfig(enabled=True)
    convexity_cfg = ConvexityHoldConfig(enabled=True)
    try:
        fp4 = _P("configs/strategies.yaml")
        if fp4.exists():
            with open(fp4, encoding="utf-8") as f:
                raw4 = _yaml.safe_load(f) or {}
            port_raw2 = (raw4.get("portfolio") or {}) if isinstance(raw4, dict) else {}
            conv_raw = (port_raw2.get("convexity_hold") or {}) if isinstance(port_raw2, dict) else {}
            if isinstance(conv_raw, dict):
                parsed_c = ConvexityHoldConfig.from_yaml(conv_raw)
                if parsed_c.enabled:
                    convexity_cfg = parsed_c
                else:
                    convexity_cfg = ConvexityHoldConfig(
                        enabled=True,
                        min_gap=parsed_c.min_gap,
                        score_drop_pct=parsed_c.score_drop_pct,
                        crisis_regimes=parsed_c.crisis_regimes,
                        w_top=parsed_c.w_top,
                        max_gross=parsed_c.max_gross,
                        skip_capacity_route=parsed_c.skip_capacity_route,
                    )
    except Exception:
        convexity_cfg = ConvexityHoldConfig(enabled=True)
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
    policy = PortfolioPolicy(sizing_config=cfg, master=_master, max_per_theme=2, max_per_family=1, lottery_config=lottery_cfg, convexity_config=convexity_cfg)
    policy.name = "portfolio.convexity_hold"  # type: ignore[attr-defined]
    policy.scores_path_independent = True
    policy.lottery_config = lottery_cfg
    policy.convexity_config = convexity_cfg

    def _score(snapshot, context):  # type: ignore[no-untyped-def]
        raw = family_model.score(snapshot, context)
        if not raw:
            return {}
        try:
            return family_canonical_scores(raw, _master)
        except Exception:
            return {}

    policy.score = _score  # type: ignore[attr-defined]
    _p16_anchor = "portfolio.convexity_hold"  # noqa: F401
    _p15_anchor2 = "portfolio.tail_concentration"  # noqa: F401
    return policy


def make_portfolio_convexity_rebalance() -> object:
    from pathlib import Path as _P

    from src.portfolio.constraints import load_rebalance_threshold

    threshold = load_rebalance_threshold(_P("configs/portfolio.yaml"))
    if abs(float(threshold)) <= 1e-12:
        raise ValueError("rebalance_threshold zero would duplicate P16")
    base = make_portfolio_convexity_hold()
    base.name = "portfolio.convexity_rebalance"  # type: ignore[attr-defined]
    base.min_rebalance_delta = float(threshold)  # type: ignore[attr-defined]
    return base


def make_portfolio_convexity_variant() -> object:
    from pathlib import Path as _P

    from src.portfolio.constraints import load_rebalance_threshold

    threshold = load_rebalance_threshold(_P("configs/portfolio.yaml"))
    base = make_portfolio_convexity_hold()
    base.name = "portfolio.convexity_variant"  # type: ignore[attr-defined]
    base.min_rebalance_delta = float(threshold)  # type: ignore[attr-defined]
    return base


def make_portfolio_lottery_rebalance() -> object:
    from pathlib import Path as _P

    from src.portfolio.constraints import load_rebalance_threshold

    threshold = load_rebalance_threshold(_P("configs/portfolio.yaml"))
    if abs(float(threshold)) <= 1e-12:
        raise ValueError("rebalance_threshold zero would duplicate P14")
    base = make_portfolio_lottery_exposure()
    base.name = "portfolio.lottery_rebalance"  # type: ignore[attr-defined]
    base.min_rebalance_delta = float(threshold)  # type: ignore[attr-defined]
    base.convexity_config = None  # type: ignore[attr-defined]
    return base
