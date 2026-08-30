from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.backtest.costs import CostConfig
from src.backtest.engine import BacktestConfig
from src.backtest.session_cache import build_session_cache
from src.core.calendar import TradingCalendar
from src.portfolio.sizing import ConfidenceSizingConfig, SizingScheme
from src.portfolio.policy import PortfolioPolicy
from tests.unit.backtest.conftest import build_engine, panel_row


def test_build_session_cache_leverage_propagation() -> None:
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 10))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0) for d in sessions])
    engine, _, filt = build_engine(panel)
    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    policy = PortfolioPolicy(sizing_config=ConfidenceSizingConfig())
    policy.name = "P11"  # type: ignore[attr-defined]

    def _score(snapshot, ctx):  # type: ignore[no-untyped-def]
        return {"069500": 1.0}

    policy.score = _score  # type: ignore[attr-defined]
    cache = build_session_cache(engine, policy, panel, config, leverage_allowed=True, inverse_allowed=False)
    assert cache.rules is not None
    assert getattr(cache.rules, "leverage_allowed", None) is True
    assert getattr(cache.rules, "inverse_allowed", None) is False
    # also test None propagation not overriding?
    cache2 = build_session_cache(engine, policy, panel, config, leverage_allowed=None)
    # when None, should not be forced True; could be UNKNOWN or None but not True forced
    # At least not assert True if original yaml UNKNOWN
    _ = cache2


@pytest.mark.parametrize("scenario_id", ["test_build_session_cache_leverage_propagation"])
def test_scenario_wrapper(scenario_id: str) -> None:
    test_build_session_cache_leverage_propagation()
