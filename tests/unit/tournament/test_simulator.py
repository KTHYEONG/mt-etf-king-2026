"""SCENARIO-06-09 SCENARIO-06-10"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import polars as pl

from src.alpha.baselines import BASELINES, RegimeGatedMomentum
from src.alpha.base import DecisionContext
from src.backtest.costs import CostConfig
from src.backtest.engine import BacktestConfig
from src.features.regime import RegimeSnapshot, RegimeState
from src.portfolio.sizing import SizingScheme
from src.core.calendar import TradingCalendar
from src.tournament.simulator import TournamentSimulator
from src.universe.tournament import TournamentRules
from tests.unit.backtest.conftest import build_engine, panel_row


def _rules() -> TournamentRules:
    return TournamentRules(
        name="test",
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 20),
        initial_capital=1_000_000_000,
        category="autonomous",
        leverage_allowed=True,
        inverse_allowed=True,
        max_weight=1.0,
        cash_allowed=True,
        sponsor_etf_only=False,
        manifest_path=None,
        issuer_whitelist=None,
        commission_bps=0.0,
        slippage_bps=0.0,
        max_order_to_adv=0.05,
        stress_grid=(0.01, 0.02, 0.05),
    )


def _synthetic_panel(sessions: list[date]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for i, d in enumerate(sessions):
        mom_a = 0.20 + i * 0.001
        mom_b = 0.10 + i * 0.001
        rows.append(panel_row(day=d, ticker="069500", close=30_000.0 + i, mom_20=mom_a, name="KODEX 200"))
        rows.append(panel_row(day=d, ticker="451060", close=20_000.0 + i, mom_20=mom_b, name="KODEX K-반도체"))
    return pl.DataFrame(rows)


def _flat_panel(sessions: list[date]) -> pl.DataFrame:
    return pl.DataFrame(
        [panel_row(day=d, ticker="069500", close=30_000.0, mom_20=0.2, name="KODEX 200") for d in sessions]
    )


def test_SCENARIO_06_09_rolling_fast_vs_slow_path() -> None:
    """SCENARIO-06-09"""
    cal = TradingCalendar()
    panel = _flat_panel(cal.sessions(date(2026, 1, 2), date(2026, 2, 28)))
    engine, cal, filt = build_engine(panel)
    config = BacktestConfig(
        start=date(2026, 1, 2),
        end=date(2026, 2, 28),
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    model = BASELINES["B0"]()

    mock_engine = MagicMock(wraps=engine)
    fast_sim = TournamentSimulator(mock_engine, cal)
    fast = fast_sim.run_rolling(model, panel, config, horizon=5, path_dependent=False)
    assert mock_engine.run.call_count == 1

    slow_mock = MagicMock(wraps=engine)
    slow_sim = TournamentSimulator(slow_mock, cal)
    slow = slow_sim.run_rolling(model, panel, config, horizon=5, path_dependent=True)
    assert slow.returns
    assert len(slow.returns) > 1
    assert slow_mock.run.call_count == len(slow.returns)

    for a, b in zip(fast.returns, slow.returns, strict=True):
        assert abs(a - b) < 1e-10


def test_SCENARIO_06_10_baselines_registry_and_engine_path() -> None:
    """SCENARIO-06-10"""
    cal = TradingCalendar()
    panel = _synthetic_panel(cal.sessions(date(2026, 1, 2), date(2026, 1, 20)))
    engine, _, filt = build_engine(panel)
    config = BacktestConfig(
        start=date(2026, 1, 2),
        end=date(2026, 1, 20),
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    daily_len: int | None = None
    for key in ("B0", "B1", "B2", "B3", "B4", "B5"):
        assert key in BASELINES
        model = BASELINES[key]()
        assert hasattr(model, "score")
        result = engine.run(model, panel, config)
        assert result.daily.height > 0
        if daily_len is None:
            daily_len = result.daily.height
        else:
            assert result.daily.height == daily_len

    gated = RegimeGatedMomentum(inner=BASELINES["B4"](), blocked=frozenset({RegimeState.STRONG_RISK_OFF}), name="B5")
    snap = pl.DataFrame([{"ticker": "069500", "mom_20": 0.2}])
    rules = _rules()
    blocked_ctx = DecisionContext(
        decision_date=date(2026, 1, 2),
        regime=RegimeSnapshot(
            as_of=date(2026, 1, 2),
            state=RegimeState.STRONG_RISK_OFF,
            score=0.0,
            components={},
        ),
        capital=1_000_000_000.0,
        held={},
        rules=rules,
    )
    assert gated.score(snap, blocked_ctx) == {}
    open_ctx = DecisionContext(
        decision_date=date(2026, 1, 2),
        regime=RegimeSnapshot(
            as_of=date(2026, 1, 2),
            state=RegimeState.NEUTRAL,
            score=0.5,
            components={"breadth": 0.5},
        ),
        capital=1_000_000_000.0,
        held={},
        rules=rules,
    )
    assert gated.score(snap, open_ctx)
