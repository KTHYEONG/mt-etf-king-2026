"""P4 CLI decomposition: backtest/_prep.py run preparation surface."""
import dataclasses

from src.cli.commands.backtest._prep import _Prep, prepare_run


def test_p4_backtest_prep_surface() -> None:
    assert callable(prepare_run)
    prep_fields = {f.name for f in dataclasses.fields(_Prep)}
    assert "b1_anchor_cache" in prep_fields
    assert "close_map" in prep_fields
    assert _Prep().b1_anchor_cache == {}


from datetime import date
from unittest.mock import patch

import polars as pl

from src.backtest.session_cache import SessionCacheRegistry, _build_open_map
from src.cli.context import BacktestContext
from src.core.calendar import TradingCalendar
from src.core.paths import DataPaths
from src.strategies.registry import STRATEGIES
from tests.unit.backtest.conftest import panel_row


def test_prepare_run_creates_registry_and_open_map_without_eager_cache(tmp_path) -> None:
    cal = TradingCalendar()
    sessions = cal.sessions(date(2024, 1, 2), date(2024, 2, 29))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0 + i)
                          for i, d in enumerate(sessions)])
    ctx = BacktestContext(
        strategy_id="sticky.mom60_raw",
        strategy=STRATEGIES["sticky.mom60_raw"](),
        start=sessions[0],
        end=sessions[-1],
        paths=DataPaths(root=tmp_path),
        calendar=cal,
        panel=panel,
        leverage_scenario="aggressive",
        eval_mode="adoption",
        protocol="grid",
        commission_bps=None,
        slippage_bps=None,
        participation=None,
    )

    # When: the run apparatus is assembled (path-dependent model, grid protocol)
    with patch("src.backtest.session_cache.build_session_cache") as eager:
        prep = prepare_run(ctx)

    # Then: no eager cases[0] cache was built
    assert eager.call_count == 0
    assert prep.shared_cache is None
    # And: a fresh empty registry is exposed for the cell loop
    assert isinstance(prep.cache_registry, SessionCacheRegistry)
    assert prep.cache_registry.builds == 0
    assert prep.cache_registry.hits == 0
    # And: the open map is prebuilt once and matches the direct build
    assert prep.open_map == _build_open_map(panel)
    assert prep.close_map is not None


def test_prepare_run_bounds_working_panel_but_master_keeps_full_history(tmp_path) -> None:
    cal = TradingCalendar()
    # Given: ticker OLD trades continuously for ~286 sessions; the request window starts 40
    # sessions before the data's end, i.e. far into OLD's history
    all_sessions = cal.sessions(date(2023, 1, 2), date(2024, 2, 29))
    start = all_sessions[-40]
    end = all_sessions[-1]
    panel = pl.DataFrame(
        [panel_row(day=d, ticker="OLD", close=30000.0 + i) for i, d in enumerate(all_sessions)]
    )
    ctx = BacktestContext(
        strategy_id="baseline.mom20_top1",
        strategy=STRATEGIES["baseline.mom20_top1"](),
        start=start,
        end=end,
        paths=DataPaths(root=tmp_path),
        calendar=cal,
        panel=panel,
        leverage_scenario="aggressive",
        eval_mode="adoption",
        protocol="single",
        commission_bps=3.0,
        slippage_bps=5.0,
        participation=0.01,
    )

    # When
    prep = prepare_run(ctx)

    # Then: master keeps the TRUE, ancient first_seen -- it was built from the full panel
    assert prep.master.attributes["OLD"].first_seen == all_sessions[0]
    # And: the working panel handed downstream is bounded, not the full ~286-session history
    assert prep.panel.height < panel.height
    assert prep.panel["date"].min() > all_sessions[0]
    assert prep.panel["date"].min() <= start
    assert prep.panel["date"].max() == end
    # And: close_map/open_map (built from the same bounded panel) stay consistent with it
    assert set(prep.close_map.keys()) <= set(prep.panel["date"].unique().to_list())


def test_prepare_run_full_span_unaffected_by_bounding(tmp_path) -> None:
    cal = TradingCalendar()
    # Given: the request start/end span the WHOLE panel -- there is no earlier data to trim
    sessions = cal.sessions(date(2024, 1, 2), date(2024, 2, 29))
    panel = pl.DataFrame(
        [panel_row(day=d, ticker="069500", close=30000.0 + i) for i, d in enumerate(sessions)]
    )
    ctx = BacktestContext(
        strategy_id="baseline.mom20_top1",
        strategy=STRATEGIES["baseline.mom20_top1"](),
        start=sessions[0],
        end=sessions[-1],
        paths=DataPaths(root=tmp_path),
        calendar=cal,
        panel=panel,
        leverage_scenario="aggressive",
        eval_mode="adoption",
        protocol="single",
        commission_bps=3.0,
        slippage_bps=5.0,
        participation=0.01,
    )

    # When
    prep = prepare_run(ctx)

    # Then: nothing to trim -- the working panel keeps every row
    assert prep.panel.height == panel.height
    assert prep.panel["date"].min() == sessions[0]
    assert prep.panel["date"].max() == sessions[-1]

