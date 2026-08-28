from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.engine import BacktestEngine
from src.backtest.execution import NextOpenExecution
from src.core.calendar import TradingCalendar
from src.features.builder import FeatureBuilder, FeatureConfig
from src.features.regime import RegimeConfig
from src.universe.instruments import InstrumentMaster
from src.universe.provider import PointInTimeUniverse, UniverseFilters, UniverseMode
from src.universe.taxonomy import Taxonomy


def feature_config() -> FeatureConfig:
    return FeatureConfig(
        momentum_horizons=(20,),
        ma_windows=(20,),
        breakout_windows=(20,),
        volatility_windows=(5,),
        flow_windows=(5,),
        regime=RegimeConfig(
            weights={},
            thresholds=(0.25, 0.45, 0.65, 0.85),
            breadth_floor=0.5,
            volatility_ceiling=0.025,
        ),
    )


def panel_row(
    *,
    day: date,
    ticker: str,
    close: float,
    open_: float | None = None,
    is_tradable: bool = True,
    trading_value: float = 5_000_000_000,
    mom_20: float = 0.01,
    name: str = "Test ETF",
    theme: str = "ThemeA",
) -> dict[str, object]:
    return {
        "date": day,
        "ticker": ticker,
        "name": name,
        "close": close,
        "open": close if open_ is None else open_,
        "high": close * 1.01,
        "low": close * 0.99,
        "is_tradable": is_tradable,
        "trading_value": trading_value,
        "underlying_index_name": theme,
        "mom_20": mom_20,
    }


def build_engine(
    panel: pl.DataFrame,
    *,
    warmup_sessions: int = 1,
    max_order_to_adv: float = 0.05,
) -> tuple[BacktestEngine, TradingCalendar, UniverseFilters]:
    cal = TradingCalendar()
    taxonomy = Taxonomy(rules=[])
    master = InstrumentMaster.build(panel, taxonomy, {})
    universe = PointInTimeUniverse(panel, master, cal, adv_window=1, brand_map={})
    builder = FeatureBuilder(cal, feature_config())
    execution = NextOpenExecution(cal)
    filt = UniverseFilters(
        mode=UniverseMode.STRUCTURAL,
        warmup_sessions=warmup_sessions,
        max_order_to_adv=max_order_to_adv,
    )
    return BacktestEngine(cal, universe, builder, execution), cal, filt
