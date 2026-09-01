"""Engine integration checks for execution timing and costs. SCENARIO-07P-02"""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import polars as pl

from src.alpha.baselines import BASELINES, BuyAndHoldBaseline  # noqa: F401
from src.backtest.costs import CostConfig, CostModel
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.core.calendar import TradingCalendar
from src.features.regime import RegimeSnapshot, RegimeState
from src.portfolio.sizing import SizingScheme
from tests.unit.backtest.conftest import build_engine, panel_row


def test_cost_applied_on_execution_date_not_signal_date() -> None:
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 9))
    rows = [
        panel_row(day=d, ticker="069500", close=100.0, mom_20=0.1, trading_value=50_000_000_000_000.0)
        for d in sessions
    ]
    panel = pl.DataFrame(rows)
    engine, _, filt = build_engine(panel, max_order_to_adv=1.0)
    strict_filt = replace(filt, max_order_to_adv=1.0)
    costs = CostConfig(commission_bps=100.0, slippage_bps=0.0, spread_bps=0.0)
    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=strict_filt,
        costs=costs,
    )
    result = engine.run(BuyAndHoldBaseline(ticker="069500", name="B0"), panel, config)
    daily = result.daily.sort("date")
    first_day_equity = float(daily.filter(pl.col("date") == sessions[0]).select("equity").item())
    second_day_equity = float(daily.filter(pl.col("date") == sessions[1]).select("equity").item())
    first_trade_weight = float(result.trades.sort("execution_date").select("weight").item(0, 0))
    expected_cost = CostModel(costs).charge(1_000_000_000.0 * first_trade_weight)
    assert first_day_equity == 1_000_000_000.0
    assert second_day_equity == 1_000_000_000.0 - expected_cost


def test_engine_applies_participation_cap_before_fill() -> None:
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 9))
    rows = [
        {
            "date": d,
            "ticker": "069500",
            "name": "KODEX 200",
            "close": 100.0,
            "open": 100.0,
            "is_tradable": True,
            "trading_value": 1_000_000_000.0,
            "underlying_index_name": "Index",
            "mom_20": 0.1,
        }
        for d in sessions
    ]
    panel = pl.DataFrame(rows)
    engine, _, filt = build_engine(panel, max_order_to_adv=0.01)
    strict_filt = replace(filt, max_order_to_adv=0.01)
    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=strict_filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    result = engine.run(BuyAndHoldBaseline(ticker="069500", name="B0"), panel, config)
    assert result.trades.height >= 1
    first_weight = float(result.trades.sort("execution_date").select("weight").item(0, 0))
    assert first_weight <= 0.01 + 1e-9


def test_SCENARIO_07P_02_regime_gating() -> None:  # noqa: N802
    """SCENARIO-07P-02"""
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 13))
    rows = [panel_row(day=d, ticker="069500", close=30000.0, mom_20=0.2) for d in sessions] + [
        panel_row(day=d, ticker="451060", close=20000.0, mom_20=0.1) for d in sessions
    ]
    panel = pl.DataFrame(rows)
    engine, cal2, filt = build_engine(panel, warmup_sessions=1, max_order_to_adv=1.0)
    from dataclasses import replace

    filt2 = replace(filt, max_order_to_adv=1.0)
    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1e9, scheme=SizingScheme.TOP1, k=1, filters=filt2, costs=CostConfig(0, 0, 0))
    # regimes mapping every session to STRONG_RISK_OFF
    regimes = {d: RegimeSnapshot(as_of=d, state=RegimeState.STRONG_RISK_OFF, score=0.0, components={}) for d in sessions}
    engine_gated = BacktestEngine(cal2, engine.universe, engine.features, engine.execution, regimes=regimes)
    # four-arg construction still works
    _engine4 = BacktestEngine(cal2, engine.universe, engine.features, engine.execution)
    assert _engine4.regimes is None
    res_b5 = engine_gated.run(BASELINES["B5"](), panel, config)
    assert res_b5.trades.height == 0
    res_b4 = engine.run(BASELINES["B4"](), panel, config)
    assert res_b4.trades.height >= 1


def test_engine_run_uses_cli_leverage_override() -> None:
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.sizing import ConfidenceSizingConfig
    from src.universe.instruments import InstrumentAttributes, InstrumentMaster
    from src.universe.taxonomy import Taxonomy

    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 9))
    rows = [
        {
            "date": d,
            "ticker": "T1",
            "name": "KODEX 200",
            "close": 30000.0,
            "open": 30000.0,
            "high": 30300.0,
            "low": 29700.0,
            "is_tradable": True,
            "trading_value": 5_000_000_000.0,
            "underlying_index_name": "KOSPI 200",
            "mom_20": 0.5,
        }
        for d in sessions
    ] + [
        {
            "date": d,
            "ticker": "T2",
            "name": "KODEX 레버리지",
            "close": 20000.0,
            "open": 20000.0,
            "high": 20200.0,
            "low": 19800.0,
            "is_tradable": True,
            "trading_value": 5_000_000_000.0,
            "underlying_index_name": "KOSPI 200",
            "mom_20": 0.1,
        }
        for d in sessions
    ]
    panel = pl.DataFrame(rows)
    taxonomy = Taxonomy(rules=[])
    master = InstrumentMaster.build(panel, taxonomy, {})
    a = master.attributes["T1"]
    b = master.attributes["T2"]
    fk = a.leverage_family_key
    master2 = InstrumentMaster(
        attributes={
            "T1": a,
            "T2": InstrumentAttributes(
                ticker=b.ticker,
                name=b.name,
                issuer=b.issuer,
                leverage_multiple=2,
                leverage_family_key=fk,
                is_synthetic=False,
                is_hedged=False,
                is_active=True,
                index_key=a.index_key,
                theme=a.theme,
                first_seen=a.first_seen,
                last_seen=a.last_seen,
                left_censored=a.left_censored,
                confidence=a.confidence,
            ),
        },
        panel_start=master.panel_start,
    )
    from src.universe.provider import PointInTimeUniverse

    engine, cal2, filt = build_engine(panel, warmup_sessions=1, max_order_to_adv=1.0)
    universe2 = PointInTimeUniverse(panel, master2, cal2, adv_window=1, brand_map={})
    filt2 = replace(filt, max_order_to_adv=1.0)
    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1e9,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt2,
        costs=CostConfig(0, 0, 0),
    )
    regimes = {d: RegimeSnapshot(as_of=d, state=RegimeState.RISK_ON, score=0.9, components={}) for d in sessions}
    policy = PortfolioPolicy(sizing_config=ConfidenceSizingConfig(), master=master2, state_enabled=False)
    policy.name = "P11"  # type: ignore[attr-defined]

    def _score(snapshot, ctx):  # type: ignore[no-untyped-def]
        out: dict[str, float] = {}
        for row in snapshot.iter_rows(named=True):
            t = str(row.get("ticker"))
            v = row.get("mom_20")
            if v is None:
                continue
            out[t] = float(v)
        return out

    policy.score = _score  # type: ignore[attr-defined]
    engine_lev = BacktestEngine(
        cal2,
        universe2,
        engine.features,
        engine.execution,
        regimes=regimes,
        leverage_allowed=True,
        inverse_allowed=False,
    )
    result = engine_lev.run(policy, panel, config)
    traded = {str(t) for t in result.trades.select("ticker").to_series().to_list()} if result.trades.height > 0 else set()
    assert "T2" in traded
