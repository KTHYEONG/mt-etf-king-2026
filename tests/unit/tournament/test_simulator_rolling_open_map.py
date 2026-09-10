from datetime import date
from unittest.mock import patch

import polars as pl

from src.backtest.costs import CostConfig
from src.backtest.engine import BacktestConfig
from src.backtest.session_cache import _build_open_map, build_close_map
from src.core.calendar import TradingCalendar
from src.portfolio.sizing import SizingScheme
from src.tournament.simulator import TournamentSimulator
from tests.unit.backtest.conftest import build_engine, panel_row


def test_run_rolling_forwards_open_map_to_engine_run() -> None:
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 2, 13))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0 + 100 * i)
                          for i, d in enumerate(sessions)])
    engine, _cal, filt = build_engine(panel)
    simulator = TournamentSimulator(engine, cal)

    class _Model:
        name = "unit.rolling.openmap"
        path_dependent = False

        def score(self, snapshot, ctx):
            return {"069500": 1.0}

    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000_000.0,
                            scheme=SizingScheme.TOP1, k=1, filters=filt,
                            costs=CostConfig(0.0, 0.0, 0.0))
    open_map = _build_open_map(panel)

    # When: run_rolling is given a prebuilt open map
    with patch("src.backtest.session_cache._build_open_map", wraps=_build_open_map) as builder:
        rolling = simulator.run_rolling(
            _Model(), panel, config, horizon=5, path_dependent=False,
            close_map=build_close_map(panel), open_map=open_map,
        )

    # Then: no engine-side rebuild happened and the rolling result is populated
    assert builder.call_count == 0
    assert len(rolling.returns) > 0
    assert len(rolling.returns) == len(rolling.starts)

