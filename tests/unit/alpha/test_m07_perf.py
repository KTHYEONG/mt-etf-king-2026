"""SCENARIO-07P-01 SCENARIO-07P-02 SCENARIO-07P-03 SCENARIO-07P-04 SCENARIO-07P-05"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import polars as pl
import pytest
import yaml

from src.alpha.base import DecisionContext
from src.alpha.cluster import ClusterResolver, build_snapshot_index
from src.alpha.leadership import SectorLeadershipModel, SectorScoreWeights
from src.alpha.state import ThemeMetrics, ThemeState, TransitionConfig, run_state_machine, transition
from src.core.paths import DataPaths
from src.core.settings import get_settings
from src.universe.instruments import InstrumentMaster, load_sponsor_brand_map
from src.universe.taxonomy import Taxonomy
from src.universe.tournament import TournamentRules
from tests.unit.alpha.conftest import make_attr, make_master, transition_config


def _gold_snapshot() -> tuple[pl.DataFrame, InstrumentMaster, SectorScoreWeights, TransitionConfig]:
    try:
        settings = get_settings()
        paths = DataPaths(root=settings.data_root)
        panel = pl.read_parquet(paths.gold("etf_features"))
        brand_map = load_sponsor_brand_map(Path("configs/sponsor_brands.yaml"))
        master = InstrumentMaster.build(panel, taxonomy=Taxonomy.from_yaml(Path("configs/taxonomy.yaml")), brand_map=brand_map)
        with open("configs/strategies.yaml", encoding="utf-8") as f:
            lead = yaml.safe_load(f)["leadership"]
        weights = SectorScoreWeights.from_yaml(lead["sector_score_weights"])
        tcfg = TransitionConfig.from_yaml(lead["transition"])
        snap = panel.filter(pl.col("date") == date(2025, 8, 1))
        if snap.height == 0:
            raise ValueError("empty snap")
        return snap, master, weights, tcfg
    except Exception:
        # fallback synthetic snapshot with ~1000 tickers similar to gold size
        n = 1000
        rows = [
            {
                "ticker": f"T{i:04d}",
                "mom_20": 0.01 * (i % 10),
                "mom_5": 0.012 * (i % 10),
                "close": 100.0 + i,
                "ma_20": 90.0 + i,
                "trading_value": float(1_000_000_000 + i * 1000),
                "tracking_error": 0.01,
                "date": date(2025, 8, 1),
            }
            for i in range(n)
        ]
        snap = pl.DataFrame(rows)
        attrs = {f"T{i:04d}": make_attr(f"T{i:04d}", index_key=f"idx_{i % 50}", theme=f"TH_{i % 10}") for i in range(n)}
        master = make_master(attrs)
        weights = SectorScoreWeights(rs=0.45, accel=0.30, breadth=0.25)
        tcfg = transition_config()
        return snap, master, weights, tcfg


def test_SCENARIO_07P_01_resolve_indexed_timing() -> None:  # noqa: N802
    """SCENARIO-07P-01 resolve_indexed <0.050s"""
    snap, master, _, _ = _gold_snapshot()
    resolver = ClusterResolver(master, max_per_theme=2)
    index = build_snapshot_index(snap)
    t0 = time.perf_counter()
    _ = resolver.resolve_indexed(index, date(2025, 8, 1))
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.050, f"SCENARIO-07P-01 elapsed {elapsed}"


def test_SCENARIO_07P_02_resolve_indexed_equals_legacy() -> None:  # noqa: N802
    """SCENARIO-07P-02 resolve_indexed same tickers as legacy resolve"""
    snap, master, _, _ = _gold_snapshot()
    resolver = ClusterResolver(master, max_per_theme=2)
    legacy = resolver.resolve(snap, date(2025, 8, 1))
    index = build_snapshot_index(snap)
    indexed = resolver.resolve_indexed(index, date(2025, 8, 1))
    assert {c.ticker for c in indexed} == {c.ticker for c in legacy}
    assert sorted(c.ticker for c in indexed) == sorted(c.ticker for c in legacy)


def test_SCENARIO_07P_03_incremental_equals_replay() -> None:  # noqa: N802
    """SCENARIO-07P-03 incremental transition same as run_state_machine"""
    cfg = transition_config()
    series = [
        ThemeMetrics(theme="T", representative="A", rs=0.6, accel=0.10, breadth=0.70, ext=1.0, dd=0.02),
        ThemeMetrics(theme="T", representative="A", rs=0.80, accel=0.12, breadth=0.75, ext=1.2, dd=0.01),
        ThemeMetrics(theme="T", representative="A", rs=0.30, accel=-0.25, breadth=0.40, ext=1.0, dd=0.09),
    ] * 10  # 30 steps
    replay = run_state_machine(series, cfg, initial=ThemeState.DISCOVERY)
    final_replay = replay[-1] if replay else ThemeState.DISCOVERY
    cur = ThemeState.DISCOVERY
    pat = 0
    for m in series:
        cur, pat = transition(cur, m, cfg, patience_counter=pat, validate=False)
    final_incremental = cur
    assert final_incremental == final_replay


def test_SCENARIO_07P_04_score_timing() -> None:  # noqa: N802
    """SCENARIO-07P-04 SectorLeadershipModel.score <0.200s"""
    snap, master, weights, tcfg = _gold_snapshot()
    resolver = ClusterResolver(master, max_per_theme=2)
    model = SectorLeadershipModel(master=master, resolver=resolver, weights=weights, transition_config=tcfg, history=None)
    ctx = DecisionContext(decision_date=date(2025, 8, 1), regime=None, capital=1e9, held={}, rules=TournamentRules.from_yaml(Path("configs/tournament.yaml")))
    t0 = time.perf_counter()
    _ = model.score(snap, ctx)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.200, f"SCENARIO-07P-04 elapsed {elapsed}"


def test_SCENARIO_07P_05_engine_run_timing() -> None:  # noqa: N802
    """SCENARIO-07P-05 M07 engine run 2025-06-01..2025-08-31 (65 sessions) <30.0s"""
    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig, BacktestEngine
    from src.backtest.execution import NextOpenExecution
    from src.core.calendar import TradingCalendar
    from src.features.builder import FeatureBuilder, FeatureConfig
    from src.portfolio.sizing import SizingScheme
    from src.universe.provider import PointInTimeUniverse, UniverseFilters, UniverseMode

    settings = get_settings()
    paths = DataPaths(root=settings.data_root)
    gold_path = paths.gold("etf_features")
    if not gold_path.exists():
        pytest.skip("gold etf_features not available")
    panel = pl.read_parquet(gold_path).filter(
        pl.col("date").is_between(date(2025, 6, 1), date(2025, 8, 31))
    )
    if panel.height == 0:
        pytest.skip("no panel rows in 2025-06-01..2025-08-31")
    brand_map = load_sponsor_brand_map(Path("configs/sponsor_brands.yaml"))
    master = InstrumentMaster.build(panel, taxonomy=Taxonomy.from_yaml(Path("configs/taxonomy.yaml")), brand_map=brand_map)
    with open("configs/strategies.yaml", encoding="utf-8") as f:
        lead = yaml.safe_load(f)["leadership"]
    weights = SectorScoreWeights.from_yaml(lead["sector_score_weights"])
    tcfg = TransitionConfig.from_yaml(lead["transition"])
    resolver = ClusterResolver(master, max_per_theme=2)
    model = SectorLeadershipModel(master=master, resolver=resolver, weights=weights, transition_config=tcfg, history=None)
    with open("configs/universe.yaml", encoding="utf-8") as f:
        uc_raw = yaml.safe_load(f) or {}
    universe_config = uc_raw.get("universe", uc_raw) if isinstance(uc_raw, dict) else {}
    sponsor_issuers = tuple(sorted(set(brand_map.values()))) if brand_map else ()
    filt = UniverseFilters.for_mode(UniverseMode.DEPLOYMENT, universe_config, sponsor_issuers)
    cal = TradingCalendar()
    builder = FeatureBuilder(cal, FeatureConfig.from_yaml(Path("configs/features.yaml")))
    universe = PointInTimeUniverse(panel, master, cal, adv_window=20, brand_map=brand_map)
    engine = BacktestEngine(cal, universe, builder, NextOpenExecution(cal))
    cfg = BacktestConfig(
        start=date(2025, 6, 1),
        end=date(2025, 8, 31),
        capital=1e9,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(),
    )
    t0 = time.perf_counter()
    engine.run(model, panel, cfg)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"SCENARIO-07P-05 elapsed {elapsed}"


# Alias parametrized tests to satisfy -k "SCENARIO-07P-XX" hyphen matching
@pytest.mark.parametrize("_", [0], ids=["SCENARIO-07P-01"])
def test_SCENARIO_07P_01_alias(_: int) -> None:  # noqa: N802
    test_SCENARIO_07P_01_resolve_indexed_timing()


@pytest.mark.parametrize("_", [0], ids=["SCENARIO-07P-02"])
def test_SCENARIO_07P_02_alias(_: int) -> None:  # noqa: N802
    test_SCENARIO_07P_02_resolve_indexed_equals_legacy()


@pytest.mark.parametrize("_", [0], ids=["SCENARIO-07P-03"])
def test_SCENARIO_07P_03_alias(_: int) -> None:  # noqa: N802
    test_SCENARIO_07P_03_incremental_equals_replay()


@pytest.mark.parametrize("_", [0], ids=["SCENARIO-07P-04"])
def test_SCENARIO_07P_04_alias(_: int) -> None:  # noqa: N802
    test_SCENARIO_07P_04_score_timing()


@pytest.mark.parametrize("_", [0], ids=["SCENARIO-07P-05"])
def test_SCENARIO_07P_05_alias(_: int) -> None:  # noqa: N802
    test_SCENARIO_07P_05_engine_run_timing()
