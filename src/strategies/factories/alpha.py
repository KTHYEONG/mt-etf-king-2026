# mypy: ignore-errors
# ruff: noqa
"""Sector-leadership and family-intensity alpha strategy factories (P3 split of src/alpha/baselines.py)."""

from __future__ import annotations

from src.alpha.leadership import SectorLeadershipModel
from src.alpha.state import TransitionConfig, transition  # noqa: F401
from src.universe.instruments import InstrumentMaster


def make_alpha_sector_leadership() -> SectorLeadershipModel:
    # lazy wiring for M07; requires master and configs - fallback to empty master for registry test
    import yaml

    from src.alpha.cluster import ClusterResolver
    from src.alpha.leadership import SectorScoreWeights
    from src.core.config import config_path as _resolve_config_path

    strat_path = _resolve_config_path("strategies")
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


def make_alpha_family_intensity() -> object:
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.intensity import FamilyIntensityConfig, FamilyIntensityModel

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

        except Exception:
            _IM = None  # type: ignore[assignment,misc]  # noqa: N806
        from src.strategies.factories._shared import load_sponsor_context

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
