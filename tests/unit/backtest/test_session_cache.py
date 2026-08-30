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
    policy.name = "P08"  # type: ignore[attr-defined]

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


@pytest.mark.parametrize("scenario_id", ["SCENARIO-PERF-02"])
def test_SCENARIO_PERF_wrapper_cache(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-PERF-02":
        test_SCENARIO_PERF_02_close_map_once()
