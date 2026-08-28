from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.core.calendar import get_calendar
from src.features.builder import FeatureBuilder, FeatureConfig
from src.universe.provider import UniverseFilters, UniverseMode, UniverseSnapshot
from tests.unit.features.conftest import make_etf_panel, session_dates


@pytest.fixture(name="feature_config")
def fixture_feature_config() -> FeatureConfig:
    return FeatureConfig.from_yaml(Path("configs/features.yaml"))


@pytest.fixture(name="feature_builder")
def fixture_feature_builder(feature_config: FeatureConfig) -> FeatureBuilder:
    return FeatureBuilder(get_calendar(), feature_config)


def test_scenario_05_10_config_yaml_and_snapshot(feature_builder: FeatureBuilder, feature_config: FeatureConfig) -> None:
    """SCENARIO-05-10"""
    assert feature_config.momentum_horizons == (3, 5, 10, 20, 40, 60)
    assert feature_config.warmup_sessions(margin=20) == 80

    sessions = session_dates(date(2026, 1, 2), 80)
    decision = sessions[-1]
    rows: list[dict[str, object]] = []
    for ticker, growth in zip(["A", "B", "C", "D", "E"], [0.01, 0.008, 0.006, 0.004, 0.002], strict=True):
        price = 10_000.0
        for d in sessions:
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "close": price,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "nav": price * 0.999,
                    "shares_outstanding": 1_000_000,
                    "net_assets": int(price * 1_000_000),
                    "trading_value": 1_000_000_000,
                }
            )
            price *= 1.0 + growth / len(sessions)
    panel = pl.DataFrame(rows)
    built = feature_builder.build_panel(panel, decision)
    universe = UniverseSnapshot(
        as_of=decision,
        mode=UniverseMode.DEPLOYMENT,
        tickers=("A", "B", "C"),
        dropped={},
        filters=UniverseFilters(warmup_sessions=80),
    )
    snap = feature_builder.snapshot(built, universe)
    assert snap.height == 3
    assert snap.select(pl.col("date").unique()).to_series().to_list() == [decision]

    two_tickers = built.filter(pl.col("ticker").is_in(["A", "B"]))
    two_universe = UniverseSnapshot(
        as_of=decision,
        mode=UniverseMode.DEPLOYMENT,
        tickers=("A", "B"),
        dropped={},
        filters=UniverseFilters(warmup_sessions=80),
    )
    restricted = feature_builder.snapshot(two_tickers, two_universe)
    rs = restricted.sort("ticker").select("mom_20_rs").to_series().to_list()
    assert rs == [1.0, 0.0]


@pytest.mark.parametrize("offset", list(range(1, 31)))
def test_scenario_05_09_shift_invariance(feature_builder: FeatureBuilder, offset: int) -> None:
    """SCENARIO-05-09"""
    sessions = session_dates(date(2025, 1, 2), 120)
    tickers = [f"T{i}" for i in range(6)]
    full_panel = make_etf_panel(sessions, tickers, growth=0.002)
    max_date = sessions[-1]
    decision = sessions[-offset]
    full_slice = feature_builder.build_panel(full_panel, max_date).filter(pl.col("date") == decision)
    truncated = feature_builder.build_panel(full_panel.filter(pl.col("date") <= decision), decision).filter(
        pl.col("date") == decision
    )
    key_cols = {"date", "ticker"}
    feature_cols = [c for c in full_slice.columns if c not in key_cols]
    for col in feature_cols:
        if full_slice[col].dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            continue
        full_vals = full_slice.sort("ticker").select(col).to_series().to_list()
        trunc_vals = truncated.sort("ticker").select(col).to_series().to_list()
        for fv, tv in zip(full_vals, trunc_vals, strict=True):
            if fv is None and tv is None:
                continue
            assert fv == pytest.approx(tv, abs=1e-12, rel=1e-12)
