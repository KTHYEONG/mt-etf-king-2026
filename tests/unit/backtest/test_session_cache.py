"""SCENARIO-PERF-02"""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import polars as pl
import pytest

from src.backtest.costs import CostConfig
from src.backtest.engine import BacktestConfig
from src.backtest.session_cache import SessionInputs, build_close_map, build_session_cache
from src.core.calendar import TradingCalendar
from src.portfolio.policy import PortfolioPolicy
from src.portfolio.sizing import ConfidenceSizingConfig, SizingScheme
from src.tournament.simulator import simulate_window_from_cache
from tests.unit.backtest.conftest import build_engine, panel_row


def test_SCENARIO_PERF_02_close_map_once() -> None:  # noqa: N802
    """SCENARIO-PERF-02"""
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 2, 15))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0 + i) for i, d in enumerate(sessions)])
    engine, _, filt = build_engine(panel)
    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig(0.0, 0.0, 0.0))
    policy = PortfolioPolicy(sizing_config=ConfidenceSizingConfig())
    policy.name = "portfolio.momentum_policy"  # type: ignore[attr-defined]

    def _score(snapshot, ctx):
        return {"069500": 1.0}

    policy.score = _score  # type: ignore[attr-defined]

    # test build_close_map returns dict keyed by date with ticker->float
    cmap = build_close_map(panel)
    assert isinstance(cmap, dict)
    assert sessions[0] in cmap
    assert "069500" in cmap[sessions[0]]
    assert isinstance(cmap[sessions[0]]["069500"], float)

    # instrumented call count <=1 for cache+2 windows
    with patch("src.backtest.session_cache.build_close_map", wraps=build_close_map) as mocked:
        cache = build_session_cache(engine, policy, panel, config)
        assert isinstance(cache, SessionInputs)
        assert mocked.call_count == 1
        # simulate 2 overlapping windows should not rebuild
        simulate_window_from_cache(policy, cache, 0, 5, 1_000_000_000.0, filt, CostConfig(0.0, 0.0, 0.0))
        simulate_window_from_cache(policy, cache, 1, 5, 1_000_000_000.0, filt, CostConfig(0.0, 0.0, 0.0))
        assert mocked.call_count == 1, f"rebuild count {mocked.call_count}"


def test_build_close_map_preserves_ticker_float_mapping() -> None:
    from datetime import date
    import polars as pl
    from src.backtest.session_cache import build_close_map
    d0 = date(2026, 1, 2)
    d1 = date(2026, 1, 5)
    panel = pl.DataFrame(
        [
            {"date": d0, "ticker": "069500", "close": 100.0},
            {"date": d0, "ticker": "114800", "close": 50.5},
            {"date": d1, "ticker": "069500", "close": None},
            {"date": d1, "ticker": "114800", "close": 51.0},
        ]
    )
    cmap = build_close_map(panel)
    assert cmap[d0]["069500"] == 100.0
    assert cmap[d0]["114800"] == 50.5
    assert "069500" not in cmap.get(d1, {})
    assert cmap[d1]["114800"] == 51.0


def test_build_session_cache_falls_back_when_session_grid_fails() -> None:
    from unittest.mock import patch

    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 9))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0) for d in sessions])
    engine, _, filt = build_engine(panel)
    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig(0.0, 0.0, 0.0))

    class _Model:
        name = "portfolio.momentum_policy"
        scores_path_independent = True

        def score(self, snapshot, ctx):
            return {"069500": 1.0}

    with patch("src.backtest.session_cache.resolve_session_grid", side_effect=RuntimeError("grid fail")):
        cache = build_session_cache(engine, _Model(), panel, config)
    assert len(cache.dates) > 0


def test_build_session_cache_returns_empty_when_calendar_unavailable() -> None:
    from unittest.mock import patch

    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 9))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0) for d in sessions])
    engine, _, filt = build_engine(panel)
    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig(0.0, 0.0, 0.0))

    class _Model:
        name = "portfolio.momentum_policy"
        scores_path_independent = True

        def score(self, snapshot, ctx):
            return {"069500": 1.0}

    engine.calendar.sessions = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no sessions"))  # type: ignore[method-assign]
    with patch("src.backtest.session_cache.resolve_session_grid", side_effect=RuntimeError("grid fail")):
        cache = build_session_cache(engine, _Model(), panel, config)
    assert cache.dates == ()


@pytest.mark.parametrize("scenario_id", ["SCENARIO-PERF-02"])
def test_SCENARIO_PERF_wrapper_cache(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-PERF-02":
        test_SCENARIO_PERF_02_close_map_once()


def test_session_inputs_propagates_engine_championship_sleeves() -> None:
    from datetime import date

    import polars as pl

    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig
    from src.backtest.session_cache import build_session_cache
    from src.portfolio.sizing import SizingScheme
    from tests.unit.backtest.conftest import build_engine, panel_row


    from src.core.calendar import TradingCalendar

    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 9))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0) for d in sessions])
    engine, _, filt = build_engine(panel)
    engine.championship_sleeves = {sessions[0]: "LOTTERY_ON"}  # type: ignore[attr-defined]
    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    model = type("M", (), {"name": "m", "scores_path_independent": True})()
    cache = build_session_cache(engine, model, panel, config)
    assert cache.championship_sleeves == {sessions[0]: "LOTTERY_ON"}




from src.backtest.session_cache import _build_open_map


def test_build_open_map_matches_row_values() -> None:
    d0 = date(2026, 1, 2)
    d1 = date(2026, 1, 5)
    panel = pl.DataFrame(
        [
            {"date": d0, "ticker": "A", "open": 100.0},
            {"date": d0, "ticker": "B", "open": 200.0},
            {"date": d1, "ticker": "A", "open": 110.0},
            {"date": d1, "ticker": "B", "open": None},
        ]
    )

    omap = _build_open_map(panel)

    assert omap[d0]["A"] == 100.0
    assert omap[d0]["B"] == 200.0
    assert omap[d1]["A"] == 110.0
    assert "B" not in omap[d1]





def test_build_open_map_empty_panel_returns_empty_dict() -> None:
    assert _build_open_map(pl.DataFrame({"date": [], "ticker": [], "open": []})) == {}
    assert _build_open_map(pl.DataFrame({"date": [], "ticker": []})) == {}



from dataclasses import replace

from src.backtest.session_cache import session_cache_key
from src.universe.provider import UniverseFilters, UniverseMode


class _Model:
    name = "unit.model"


def _config(filters: UniverseFilters, costs: CostConfig) -> BacktestConfig:
    return BacktestConfig(
        start=date(2024, 1, 2),
        end=date(2024, 3, 29),
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filters,
        costs=costs,
    )


def test_session_cache_key_excludes_costs_and_includes_filters() -> None:
    model = _Model()
    filt = UniverseFilters(mode=UniverseMode.DEPLOYMENT, capital=1_000_000_000,
                           max_position_weight=1.0, max_order_to_adv=0.01)

    # Given/When: only the cost parameters differ
    key_cheap = session_cache_key(model, _config(filt, CostConfig(0.0, 0.0, 0.0)))
    key_pricey = session_cache_key(model, _config(filt, CostConfig(15.0, 20.0, 0.0)))

    # Then: the cache is reusable across the whole cost axis
    assert key_cheap == key_pricey

    # And: every participation-relevant filter field changes the key
    base = _config(filt, CostConfig(3.0, 5.0, 0.0))
    key_base = session_cache_key(model, base)
    for changed in (
        replace(filt, max_order_to_adv=0.05),
        replace(filt, score_max_order_to_adv=0.05),
        replace(filt, capital=500_000_000),
        replace(filt, max_position_weight=0.8),
        replace(filt, warmup_sessions=40),
        replace(filt, adv_window=60),
        replace(filt, mode=UniverseMode.STRUCTURAL),
    ):
        assert session_cache_key(model, _config(changed, CostConfig(3.0, 5.0, 0.0))) != key_base

    # And: leverage scenario and model identity are part of the key
    assert session_cache_key(model, base, leverage_allowed=True) != session_cache_key(model, base, leverage_allowed=False)

    class _Other:
        name = "unit.other"

    assert session_cache_key(_Other(), base) != key_base
    # And: the key must be hashable (it is used as a dict key)
    assert isinstance(hash(key_base), int)





from src.backtest.session_cache import SessionCacheRegistry


def test_session_cache_registry_reuses_across_cost_axis_only() -> None:
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 2, 13))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0 + i)
                          for i, d in enumerate(sessions)])
    engine, _cal, filt = build_engine(panel)

    class _Model:
        name = "unit.registry"

        def score(self, snapshot, ctx):
            return {"069500": 1.0}

    model = _Model()
    registry = SessionCacheRegistry()

    def _cfg(participation: float, commission: float) -> BacktestConfig:
        return BacktestConfig(
            start=sessions[0], end=sessions[-1], capital=1_000_000_000.0,
            scheme=SizingScheme.TOP1, k=1,
            filters=replace(filt, max_order_to_adv=participation),
            costs=CostConfig(commission, 5.0, 0.0),
        )

    # When: a 2-participation x 3-cost grid is walked cell by cell
    results = {}
    for participation in (0.01, 0.05):
        for commission in (0.0, 3.0, 15.0):
            results[(participation, commission)] = registry.get_or_build(
                engine, model, panel, _cfg(participation, commission)
            )

    # Then: one build per participation, every cost variant a hit
    assert registry.builds == 2
    assert registry.hits == 4
    assert all(isinstance(v, SessionInputs) for v in results.values())
    # And: same-participation cells share the identical object...
    assert results[(0.01, 0.0)] is results[(0.01, 3.0)]
    assert results[(0.01, 0.0)] is results[(0.01, 15.0)]
    # ...while a different participation never reuses it
    assert results[(0.05, 0.0)] is not results[(0.01, 0.0)]
    assert results[(0.05, 0.0)] is results[(0.05, 15.0)]




import pytest



def test_session_cache_registry_propagates_build_failure() -> None:
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 30))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0) for d in sessions])
    engine, _cal, filt = build_engine(panel)

    class _Model:
        name = "unit.fail"

        def score(self, snapshot, ctx):
            return {}

    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000_000.0,
                            scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig(0.0, 0.0, 0.0))
    registry = SessionCacheRegistry()

    # Given: the underlying builder blows up
    with patch("src.backtest.session_cache.build_session_cache", side_effect=RuntimeError("panel broken")):  # noqa: SIM117
        with pytest.raises(RuntimeError, match="panel broken"):
            registry.get_or_build(engine, _Model(), panel, config)

    # Then: nothing was cached and no counter was credited
    assert registry.builds == 0
    assert registry.hits == 0

    # And: a subsequent healthy call still builds
    cache = registry.get_or_build(engine, _Model(), panel, config)
    assert cache is not None
    assert registry.builds == 1

