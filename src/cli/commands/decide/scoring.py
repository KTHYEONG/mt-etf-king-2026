"""Decide scoring helpers (P4 decomposition of src/cli/_impl.py)."""

from __future__ import annotations

from datetime import date

import polars as pl

from src.core.calendar import get_calendar
from src.core.config import config_path
from src.core.paths import DataPaths
from src.data.panel import BACKTEST_PANEL_COLUMNS, load_backtest_panel


def _load_panel_for_backtest(paths: DataPaths, cal: object) -> pl.DataFrame | None:

    # Delegate to load_backtest_panel with BACKTEST_PANEL_COLUMNS (cal may be unused but signature preserved)
    # wiring anchor uses load_backtest_panel
    return load_backtest_panel(paths, columns=BACKTEST_PANEL_COLUMNS)


def _scores_from_deployment_universe(panel: object, decision_date: date) -> dict[str, float]:
    """Build mom scores for tickers admitted in deployment universe at decision_date."""
    import polars as pl

    from src.universe.instruments import InstrumentMaster, load_sponsor_brand_map
    from src.universe.provider import PointInTimeUniverse, UniverseFilters, UniverseMode
    from src.universe.taxonomy import Taxonomy

    if not isinstance(panel, pl.DataFrame) or panel.height == 0:
        return {}
    cal = get_calendar()
    try:
        brand_map = load_sponsor_brand_map(config_path("sponsor_brands"))
    except Exception:
        brand_map = {}
    try:
        taxonomy = Taxonomy.from_yaml(config_path("taxonomy"))
    except Exception:
        taxonomy = Taxonomy(rules=[])
    try:
        master = InstrumentMaster.build(panel, taxonomy, brand_map)
    except Exception:
        from src.universe.instruments import Confidence, InstrumentAttributes

        attrs = {}
        for t in panel.select(pl.col("ticker")).unique().to_series().to_list():
            ts = str(t)
            attrs[ts] = InstrumentAttributes(
                ticker=ts,
                name=ts,
                issuer="삼성자산운용",
                leverage_multiple=1,
                leverage_family_key=ts,
                is_synthetic=False,
                is_hedged=False,
                is_active=True,
                index_key="KOSPI 200",
                theme="",
                first_seen=decision_date,
                last_seen=decision_date,
                left_censored=True,
                confidence=Confidence.HIGH,
            )
        master = InstrumentMaster(attributes=attrs, panel_start=decision_date)
    universe_config: dict[str, object] = {}
    try:
        import yaml

        with open(config_path("universe"), encoding="utf-8") as f:
            uc_raw = yaml.safe_load(f) or {}
        universe_config = uc_raw.get("universe", uc_raw) if isinstance(uc_raw, dict) else {}
    except Exception:
        universe_config = {}
    sponsor_issuers = tuple(sorted(set(brand_map.values()))) if brand_map else ()
    filt = UniverseFilters.for_mode(UniverseMode.DEPLOYMENT, universe_config, sponsor_issuers)
    universe = PointInTimeUniverse(panel, master, cal, adv_window=20, brand_map=brand_map)
    snap = universe.get(decision_date, filt)
    admitted = set(snap.tickers)
    if not admitted:
        return {}
    score_col = next((c for c in ("mom_20", "mom_20_rs", "close") if c in panel.columns), None)
    if score_col is None or "ticker" not in panel.columns:
        return {}
    day_panel = panel
    if "date" in panel.columns:
        try:
            day_panel = panel.filter(pl.col("date") == decision_date)
            if day_panel.height == 0:
                day_panel = panel
        except Exception:
            day_panel = panel
    scores: dict[str, float] = {}
    for row in day_panel.iter_rows(named=True):
        ticker = str(row.get("ticker"))
        if ticker not in admitted:
            continue
        val = row.get(score_col)
        if val is None:
            continue
        try:
            scores[ticker] = float(val)
        except Exception:  # noqa: S112
            continue
    return scores


__all__ = ["_load_panel_for_backtest", "_scores_from_deployment_universe"]
