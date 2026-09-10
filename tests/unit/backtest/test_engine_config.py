"""Facade fidelity for the src/backtest engine split (P5)."""

from __future__ import annotations


def test_engine_config_canonical_home() -> None:
    import src.backtest.engine as facade
    from src.backtest.engine_config import BacktestConfig, BacktestResult, build_execution_adv

    assert facade.BacktestConfig is BacktestConfig
    assert facade.BacktestResult is BacktestResult
    assert facade.build_execution_adv is build_execution_adv
    assert BacktestConfig.__module__ == "src.backtest.engine_config"


from datetime import date
from unittest.mock import patch

import polars as pl

from src.backtest.engine import BacktestEngine, build_execution_adv
from src.backtest.execution import NextOpenExecution
from src.core.calendar import TradingCalendar
from src.features.builder import FeatureBuilder, FeatureConfig
from src.features.regime import RegimeConfig
from src.universe.instruments import InstrumentMaster
from src.universe.provider import PointInTimeUniverse
from src.universe.taxonomy import Taxonomy


def test_build_execution_adv_uses_family_members_fast_path() -> None:
    cal = TradingCalendar()
    d = date(2026, 1, 2)
    panel = pl.DataFrame(
        [
            {"date": d, "ticker": "PLUS1", "name": "KODEX 200", "underlying_index_name": "\ucf54\uc2a4\ud53c 200", "close": 100.0, "open": 100.0, "trading_value": 1_000_000.0, "is_tradable": True},
            {"date": d, "ticker": "LEV2", "name": "KODEX 200\ub808\ubc84\ub9ac\uc9c0", "underlying_index_name": "\ucf54\uc2a4\ud53c 200", "close": 100.0, "open": 100.0, "trading_value": 2_000_000.0, "is_tradable": True},
        ]
    )
    master = InstrumentMaster.build(panel, Taxonomy(rules=[]), {})
    universe = PointInTimeUniverse(panel, master, cal, adv_window=1, brand_map={})
    fconfig = FeatureConfig(
        momentum_horizons=(20,), ma_windows=(20,), breakout_windows=(20,),
        volatility_windows=(20,), flow_windows=(5,),
        regime=RegimeConfig(weights={}, thresholds=(0.25, 0.45, 0.65, 0.85), breadth_floor=0.5, volatility_ceiling=0.025),
    )
    builder = FeatureBuilder(cal, fconfig)
    engine = BacktestEngine(cal, universe, builder, NextOpenExecution(cal))

    # Given: PLUS1 and LEV2 share a leverage family (same underlying index)
    fk = master.attributes["PLUS1"].leverage_family_key
    assert master.attributes["LEV2"].leverage_family_key == fk

    # When: build_execution_adv is asked for PLUS1 only
    with patch.object(InstrumentMaster, "family_members", wraps=master.family_members) as mocked:
        adv = build_execution_adv(engine, ["PLUS1"], d)

    # Then: the fast path was actually used, and family expansion still reaches LEV2
    assert mocked.called
    assert "PLUS1" in adv
    assert "LEV2" in adv


def test_build_execution_adv_fallback_scan_without_family_members() -> None:
    """R4 fallback: a master-like object without family_members uses the O(n) scan."""
    from types import SimpleNamespace

    from src.backtest.engine_config import build_execution_adv

    cal = TradingCalendar()
    d = date(2026, 1, 2)
    panel = pl.DataFrame(
        [
            {"date": d, "ticker": "PLUS1", "name": "KODEX 200", "underlying_index_name": "코스피 200", "close": 100.0, "open": 100.0, "trading_value": 1_000_000.0, "is_tradable": True},
            {"date": d, "ticker": "LEV2", "name": "KODEX 200레버리지", "underlying_index_name": "코스피 200", "close": 100.0, "open": 100.0, "trading_value": 2_000_000.0, "is_tradable": True},
        ]
    )
    master = InstrumentMaster.build(panel, Taxonomy(rules=[]), {})
    universe = PointInTimeUniverse(panel, master, cal, adv_window=1, brand_map={})
    fconfig = FeatureConfig(
        momentum_horizons=(20,), ma_windows=(20,), breakout_windows=(20,),
        volatility_windows=(20,), flow_windows=(5,),
        regime=RegimeConfig(weights={}, thresholds=(0.25, 0.45, 0.65, 0.85), breadth_floor=0.5, volatility_ceiling=0.025),
    )
    builder = FeatureBuilder(cal, fconfig)
    engine = BacktestEngine(cal, universe, builder, NextOpenExecution(cal))

    fk = master.attributes["PLUS1"].leverage_family_key
    assert master.attributes["LEV2"].leverage_family_key == fk

    # Given: a legacy master double exposing attributes but no family_members
    legacy_master = SimpleNamespace(attributes=dict(master.attributes))
    assert not hasattr(legacy_master, "family_members")
    engine.universe = SimpleNamespace(master=legacy_master, adv=universe.adv)  # type: ignore[attr-defined]

    # When
    adv = build_execution_adv(engine, ["PLUS1"], d)

    # Then: family expansion still reaches LEV2 via the fallback scan
    assert "PLUS1" in adv
    assert "LEV2" in adv

