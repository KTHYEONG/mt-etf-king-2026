from __future__ import annotations

from datetime import date

import polars as pl

from src.core.calendar import TradingCalendar
from src.universe.instruments import InstrumentMaster, load_sponsor_brand_map
from src.universe.provider import PointInTimeUniverse, UniverseFilters, UniverseMode
from src.universe.taxonomy import Taxonomy


def panel_row(
    *,
    day: date,
    ticker: str,
    name: str = "KODEX 200",
    index: str = "코스피 200",
    is_tradable: bool = True,
    close: float = 1000.0,
    trading_value: float = 2e12,
) -> dict[str, object]:
    return {
        "date": day,
        "ticker": ticker,
        "name": name,
        "underlying_index_name": index,
        "is_tradable": is_tradable,
        "close": close,
        "trading_value": trading_value,
    }


def make_panel(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def deploy_filters(
    sponsor_issuers: tuple[str, ...],
    *,
    warmup_sessions: int = 1,
    max_order_to_adv: float = 0.05,
    manifest: frozenset[str] | None = None,
) -> UniverseFilters:
    return UniverseFilters(
        mode=UniverseMode.DEPLOYMENT,
        warmup_sessions=warmup_sessions,
        max_order_to_adv=max_order_to_adv,
        issuer_whitelist=sponsor_issuers,
        manifest=manifest,
    )


def build_universe(
    panel: pl.DataFrame,
    *,
    adv_window: int = 1,
    brand_map: dict[str, str] | None = None,
) -> tuple[PointInTimeUniverse, InstrumentMaster, TradingCalendar]:
    cal = TradingCalendar()
    taxonomy = Taxonomy(rules=[])
    if brand_map is None:
        brand_map = load_sponsor_brand_map(__import__("pathlib").Path("configs/sponsor_brands.yaml"))
    master = InstrumentMaster.build(panel, taxonomy, brand_map)
    return PointInTimeUniverse(panel, master, cal, adv_window=adv_window, brand_map=brand_map), master, cal
