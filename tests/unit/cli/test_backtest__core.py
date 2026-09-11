"""P4 CLI decomposition: backtest/_core.py run loop surface."""
import dataclasses

from src.cli.commands.backtest import _core as core_mod
from src.cli.commands.backtest._core import _Cell, run_cells, run_preflight


def test_fullspan_fallback_ids_include_all_sticky_strategies_needing_exposure_artifacts() -> None:
    """Any sticky-family strategy without a path_dependent rolling.backtest must be in
    _FULLSPAN_FALLBACK_IDS, or its exposure/gross-violation metrics silently go unmeasured
    (effective_gross_mean/max stay null and gross_violation_count is never actually computed,
    only defaulted) instead of failing loudly."""
    assert "sticky.mom60_inactive_participate" in core_mod._FULLSPAN_FALLBACK_IDS
    assert "sticky.mom60_post_crash_anchor" in core_mod._FULLSPAN_FALLBACK_IDS


def test_p4_backtest_core_surface() -> None:
    assert callable(run_preflight)
    assert callable(run_cells)
    assert callable(core_mod._run_common_tail)
    assert isinstance(core_mod._PREFLIGHT_STEPS, tuple)
    cell_fields = {f.name for f in dataclasses.fields(_Cell)}
    assert "model" in cell_fields
    assert "b1_anchor_cache" in cell_fields


from dataclasses import replace
from datetime import date

import polars as pl

from src.backtest.costs import CostConfig
from src.backtest.engine import BacktestConfig
from src.backtest.session_cache import SessionCacheRegistry
from src.cli.commands.backtest._prep import _Prep
from src.cli.context import BacktestContext
from src.core.calendar import TradingCalendar
from src.core.paths import DataPaths
from src.portfolio.sizing import SizingScheme
from src.tournament.harness import iter_protocol_cases
from src.tournament.simulator import TournamentSimulator
from tests.unit.backtest.conftest import build_engine, panel_row


def test_run_cells_grid_builds_one_cache_per_participation(tmp_path) -> None:
    cal = TradingCalendar()
    sessions = cal.sessions(date(2024, 1, 2), date(2024, 3, 29))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0 + i)
                          for i, d in enumerate(sessions)])
    engine, _cal, filt = build_engine(panel)

    class _Model:
        name = "unit.grid"
        path_dependent = False

        def score(self, snapshot, ctx):
            return {"069500": 1.0}

    model = _Model()
    cases = list(iter_protocol_cases("grid", commission_bps=None, slippage_bps=None, participation=0.01))
    assert len(cases) == 36

    ctx = BacktestContext(
        strategy_id="unit.grid", strategy=model, start=sessions[0], end=sessions[-1],
        paths=DataPaths(root=tmp_path), calendar=cal, panel=panel,
        leverage_scenario="aggressive", eval_mode="adoption", protocol="grid",
        commission_bps=None, slippage_bps=None, participation=None,
    )
    prep = _Prep()
    prep.rules = None
    prep.horizon = 5
    prep.engine = engine
    prep.simulator = TournamentSimulator(engine, cal)
    prep.filt = filt
    prep.bconfig = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000_000.0,
                                  scheme=SizingScheme.TOP1, k=1, filters=filt,
                                  costs=CostConfig(3.0, 5.0, 0.0))
    prep.panel = panel
    prep.thresholds = [0.10, 0.20]
    prep.tail_weights = {0.90: 1.0}
    prep.cases = cases
    prep.is_pd = False
    prep.path_mode = "fast"
    prep.cache_registry = SessionCacheRegistry()
    prep.control_flags = [False] * len(cases)

    # When: the whole 36-cell grid is executed
    bundles = run_cells(ctx, prep, None)

    # Then: one cache per unique participation, everything else a hit
    assert len(bundles) == 36
    assert prep.cache_registry.builds == 3
    assert prep.cache_registry.hits == 33
    # And: each bundle's cache belongs to its own cell's participation
    by_participation: dict[float, set[int]] = {}
    for bundle in bundles:
        participation = float(bundle.meta["participation"])
        by_participation.setdefault(participation, set()).add(id(bundle.shared_cache))
    assert sorted(by_participation) == [0.01, 0.02, 0.05]
    for participation, cache_ids in by_participation.items():
        assert len(cache_ids) == 1, f"participation {participation} used more than one cache"
    assert len({next(iter(v)) for v in by_participation.values()}) == 3



def test_run_cells_cell_cache_failure_degrades_to_none(tmp_path) -> None:
    """A session-cache build failure must not abort the cell (pre-existing fail-soft contract)."""
    from unittest.mock import patch

    cal = TradingCalendar()
    sessions = cal.sessions(date(2024, 1, 2), date(2024, 3, 29))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0 + i)
                          for i, d in enumerate(sessions)])
    engine, _cal, filt = build_engine(panel)

    class _Model:
        name = "unit.cachefail"
        path_dependent = False

        def score(self, snapshot, ctx):
            return {"069500": 1.0}

    model = _Model()
    cases = list(iter_protocol_cases("single", commission_bps=3.0, slippage_bps=5.0, participation=0.01))
    ctx = BacktestContext(
        strategy_id="unit.cachefail", strategy=model, start=sessions[0], end=sessions[-1],
        paths=DataPaths(root=tmp_path), calendar=cal, panel=panel,
        leverage_scenario="aggressive", eval_mode="adoption", protocol="single",
        commission_bps=3.0, slippage_bps=5.0, participation=0.01,
    )
    prep = _Prep()
    prep.horizon = 5
    prep.engine = engine
    prep.simulator = TournamentSimulator(engine, cal)
    prep.filt = filt
    prep.bconfig = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000_000.0,
                                  scheme=SizingScheme.TOP1, k=1, filters=filt,
                                  costs=CostConfig(3.0, 5.0, 0.0))
    prep.panel = panel
    prep.thresholds = [0.10, 0.20]
    prep.tail_weights = {0.90: 1.0}
    prep.cases = cases
    prep.is_pd = False
    prep.path_mode = "fast"
    prep.cache_registry = SessionCacheRegistry()
    prep.control_flags = [False] * len(cases)

    # Given: the registry raises for this cell
    with patch.object(SessionCacheRegistry, "get_or_build", side_effect=RuntimeError("cache build blew up")):
        bundles = run_cells(ctx, prep, None)

    # Then: the cell still produced a bundle, with no cache attached
    assert len(bundles) == 1
    assert bundles[0].shared_cache is None
    assert bundles[0].dist is not None


def test_run_common_tail_fullspan_fallback_passes_open_map(tmp_path) -> None:
    """For _FULLSPAN_FALLBACK_IDS with no rolling.backtest, the artifact run reuses prep.open_map."""
    from unittest.mock import patch

    from src.backtest.session_cache import _build_open_map, build_close_map

    cal = TradingCalendar()
    sessions = cal.sessions(date(2024, 1, 2), date(2024, 3, 29))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0 + i)
                          for i, d in enumerate(sessions)])
    engine, _cal, filt = build_engine(panel)

    class _Model:
        name = "sticky.mom60_raw"

        def score(self, snapshot, ctx):
            return {"069500": 1.0}

    model = _Model()
    model_key = "sticky.mom60_raw"
    assert model_key in core_mod._FULLSPAN_FALLBACK_IDS

    case_config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000_000.0,
                                 scheme=SizingScheme.TOP1, k=1, filters=filt,
                                 costs=CostConfig(3.0, 5.0, 0.0))
    ctx = BacktestContext(
        strategy_id=model_key, strategy=model, start=sessions[0], end=sessions[-1],
        paths=DataPaths(root=tmp_path), calendar=cal, panel=panel,
        leverage_scenario="aggressive", eval_mode="adoption", protocol="single",
        commission_bps=3.0, slippage_bps=5.0, participation=0.01,
    )
    prep = _Prep()
    prep.horizon = 5
    prep.engine = engine
    prep.simulator = TournamentSimulator(engine, cal)
    prep.filt = filt
    prep.bconfig = case_config
    prep.panel = panel
    prep.thresholds = [0.10, 0.20]
    prep.tail_weights = {0.90: 1.0}
    prep.close_map = build_close_map(panel)
    prep.open_map = _build_open_map(panel)
    prep.control_flags = [False]
    prep.master = engine.universe.master

    rolling = prep.simulator.run_rolling(model, panel, case_config, horizon=5, path_dependent=False)
    cell = _Cell(
        model_key=model_key, eval_mode="adoption", start=sessions[0], end=sessions[-1], cal=cal,
        panel=panel, horizon=5, engine=engine, model=model, simulator=prep.simulator,
        thresholds=prep.thresholds, tail_weights=prep.tail_weights, lev_allowed=None, inv_allowed=None,
        regimes=None, close_map=prep.close_map, shared_cache=None, rolling_exposure_limits=None,
        path_mode="fast", master=prep.master, cost_cfg=CostConfig(3.0, 5.0, 0.0), participation=0.01,
        filt_case=filt, case_config=case_config, trace_sink=None,
        rolling=dataclasses.replace(rolling, backtest=None), dist=None, run_id="unit-run",
        meta={}, summary={}, control_cache=None, control_flags=prep.control_flags, cell_idx=0,
        b1_anchor_cache={}, b1_dist_cache_p16={}, b1_dist_cache_p24={},
    )

    # When: the full-span artifact fallback runs
    with patch("src.backtest.session_cache._build_open_map", wraps=_build_open_map) as builder:
        core_mod._run_common_tail(ctx, prep, model_key, cell, 0)

    # Then: the artifact run happened without rebuilding the open map
    assert cell.daily is not None
    assert builder.call_count == 0
