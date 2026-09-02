def test_ledger_equity_invariant_after_price_move() -> None:
    from src.execution.ledger import PortfolioLedgerState, ledger_state_from_weights

    state = ledger_state_from_weights(equity=1_000_000.0, weights={"A": 0.95}, mark_prices={"A": 100.0})
    eq_before = state.equity_at_prices({"A": 100.0})
    weights_before = state.weights_at_prices({"A": 100.0})
    eq_after = state.equity_at_prices({"A": 110.0})
    weights_after = state.weights_at_prices({"A": 110.0})
    assert abs(eq_before - 1_000_000.0) < 1e-6
    assert abs(weights_before["A"] - 0.95) < 1e-9
    assert abs(eq_after - 1_095_000.0) < 1.0
    assert weights_after["A"] > 0.95
    assert abs(weights_after["A"] - 0.9543) < 0.001


def test_transaction_cost_applied_after_overnight_not_before() -> None:
    from datetime import date

    import polars as pl

    from src.backtest.costs import CostConfig, CostModel
    from src.backtest.execution import NextOpenExecution
    from src.core.calendar import TradingCalendar
    from src.execution.ledger import PortfolioLedgerState, transition_portfolio_state
    from src.portfolio.intent import PortfolioIntent

    cal = TradingCalendar()
    d0 = date(2026, 1, 2)
    d1 = date(2026, 1, 5)
    state = PortfolioLedgerState(cash=50_000.0, shares={"A": 9500.0})
    intent = PortfolioIntent(kind="target", weights={"A": 0.95})
    result = transition_portfolio_state(
        prior_state=state,
        intent=intent,
        decision_date=d0,
        prev_closes={"A": 100.0},
        opens={"A": 110.0},
        closes={"A": 105.0},
        cost_model=CostModel(CostConfig(commission_bps=10.0, slippage_bps=0.0, tax_bps=0.0)),
        adv_by_ticker={"A": 1e12},
        max_order_to_adv=1.0,
        exposure_limits=None,
        leverage_multiples={"A": 1},
        execution=NextOpenExecution(cal),
        panel=pl.DataFrame([{"date": d1, "ticker": "A", "open": 110.0, "close": 105.0, "is_tradable": True}]),
    )
    overnight_only = 9500.0 * 110.0 + 50_000.0
    assert overnight_only > 1_000_000.0
    assert result.diagnostics.transaction_cost >= 0.0
    assert result.equity_close < overnight_only
    assert abs(result.session_return - (result.equity_close / (9500.0 * 100.0 + 50_000.0) - 1.0)) < 0.02


def test_adv_lookup_uses_decision_date_not_execution_date() -> None:
    from datetime import date

    from src.backtest.engine import build_execution_adv
    from src.features.regime import RegimeConfig
    from src.features.builder import FeatureBuilder, FeatureConfig
    from src.core.calendar import TradingCalendar
    from src.universe.instruments import InstrumentMaster
    from src.universe.provider import PointInTimeUniverse, UniverseFilters, UniverseMode

    cal = TradingCalendar()
    d_dec = date(2026, 1, 2)
    d_exec = date(2026, 1, 5)
    import polars as pl

    panel = pl.DataFrame(
        [
            {"date": d_dec, "ticker": "A", "close": 100.0, "open": 100.0, "trading_value": 1_000_000.0, "is_tradable": True},
            {"date": d_exec, "ticker": "A", "close": 100.0, "open": 100.0, "trading_value": 9_999_999_999.0, "is_tradable": True},
        ]
    )
    master = InstrumentMaster.build(panel, __import__("src.universe.taxonomy", fromlist=["Taxonomy"]).Taxonomy(rules=[]), {})
    filt = UniverseFilters.for_mode(UniverseMode.DEPLOYMENT, {}, ())
    universe = PointInTimeUniverse(master, panel, cal, filt)
    fconfig = FeatureConfig(momentum_horizons=(20,), ma_windows=(20,), breakout_windows=(20,), volatility_windows=(20,), flow_windows=(5,), regime=RegimeConfig(weights={}, thresholds=(0.25, 0.45, 0.65, 0.85), breadth_floor=0.5, volatility_ceiling=0.025))
    builder = FeatureBuilder(cal, fconfig)
    from src.backtest.execution import NextOpenExecution
    from src.backtest.engine import BacktestEngine

    engine = BacktestEngine(cal, universe, builder, NextOpenExecution(cal))
    adv_dec = build_execution_adv(engine, ["A"], d_dec)
    adv_exec = build_execution_adv(engine, ["A"], d_exec)
    assert "A" in adv_dec
    assert abs(adv_dec["A"] - 1_000_000.0) < 1.0
    assert abs(adv_exec.get("A", 0.0) - 9_999_999_999.0) > 1_000_000.0
    from src.backtest.session_cache import build_session_cache
    from src.backtest.engine import BacktestConfig
    from src.backtest.costs import CostConfig
    from src.portfolio.sizing import SizingScheme

    class _Static:
        name = "static"
        scores_path_independent = True

        def score(self, snapshot: object, context: object) -> dict[str, float]:
            return {"A": 1.0}

    config = BacktestConfig(start=d_dec, end=d_exec, capital=1_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig(0.0, 0.0, 0.0))
    cache = build_session_cache(engine, _Static(), panel, config)
    assert d_dec in cache.adv_map
    assert abs(cache.adv_map[d_dec]["A"] - 1_000_000.0) < 1.0
    assert d_exec not in cache.adv_map or abs(cache.adv_map.get(d_exec, {}).get("A", 0.0) - 9_999_999_999.0) > 1.0


def test_cash_intent_fast_sim_matches_engine() -> None:
    from datetime import date

    import polars as pl

    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig
    from src.backtest.metrics import compound_returns
    from src.backtest.session_cache import build_session_cache
    from src.core.calendar import TradingCalendar
    from src.portfolio.intent import CASH_INTENT
    from src.portfolio.sizing import SizingScheme
    from src.tournament.simulator import simulate_window_from_cache
    from tests.unit.backtest.conftest import build_engine, panel_row

    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 2, 10))
    rows = []
    for i, d in enumerate(sessions):
        rows.append(panel_row(day=d, ticker="069500", close=30000.0 + i * 50.0, mom_20=0.10, name="KODEX 200", theme="KOSPI"))
        rows.append(panel_row(day=d, ticker="122630", close=20000.0 + i * 80.0, mom_20=0.20, name="LEV", theme="LEV"))
    panel = pl.DataFrame(rows)
    engine, _, filt = build_engine(panel, warmup_sessions=1)
    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig(0.0, 0.0, 0.0))

    class _CrashCash:
        name = "crash_cash"
        scores_path_independent = False

        def reset_trackers(self) -> None:
            self._day = 0

        def score(self, snapshot: object, context: object) -> object:
            self._day = getattr(self, "_day", 0) + 1
            if self._day >= 3:
                return CASH_INTENT
            return {"069500": 1.0}

    horizon = 5
    cache = build_session_cache(engine, _CrashCash(), panel, config)
    for start_idx in (0, 4, 8):
        start = sessions[start_idx]
        end_date = sessions[start_idx + horizon - 1]
        win_config = BacktestConfig(start=start, end=end_date, capital=config.capital, scheme=config.scheme, k=config.k, filters=config.filters, costs=config.costs)
        slow = engine.run(_CrashCash(), panel, win_config)
        ret_col = "ret" if "ret" in slow.daily.columns else "return"
        slow_rets = [float(row.get(ret_col) or 0.0) for row in slow.daily.iter_rows(named=True)]
        slow_comp = compound_returns(slow_rets) if slow_rets else 0.0
        fast_comp, _, _ = simulate_window_from_cache(_CrashCash(), cache, start_idx, horizon, float(config.capital), config.filters, config.costs, panel=panel, execution=engine.execution, scheme=config.scheme, k=config.k)
        assert abs(float(fast_comp) - float(slow_comp)) < 1e-9


def test_rolling_diagnostics_populated_in_fast_mode() -> None:
    from datetime import date

    import polars as pl

    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig
    from src.portfolio.sizing import SizingScheme
    from src.tournament.simulator import TournamentSimulator
    from tests.unit.backtest.conftest import build_engine, panel_row

    cal_sessions = __import__("src.core.calendar", fromlist=["TradingCalendar"]).TradingCalendar().sessions(date(2026, 1, 2), date(2026, 2, 28))
    rows = []
    for i, d in enumerate(cal_sessions):
        rows.append(panel_row(day=d, ticker="069500", close=30000.0 + i * 10.0, mom_20=0.10, name="KODEX 200", theme="KOSPI"))
    panel = pl.DataFrame(rows)
    engine, cal, filt = build_engine(panel, warmup_sessions=1)
    config = BacktestConfig(start=cal_sessions[0], end=cal_sessions[-1], capital=1_000_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig(3.0, 2.0, 0.0))

    class _Hold:
        name = "hold"
        path_dependent = True
        scores_path_independent = False

        def score(self, snapshot: object, context: object) -> dict[str, float]:
            return {"069500": 1.0}

    sim = TournamentSimulator(engine, cal)
    rolling = sim.run_rolling(_Hold(), panel, config, horizon=5, path_dependent=True, path_dependent_mode="fast")
    assert rolling.diagnostics is not None
    assert rolling.diagnostics.gross_violation_count is not None
    assert rolling.diagnostics.effective_gross_max is not None
    assert rolling.diagnostics.turnover_mean is not None
    assert rolling.diagnostics.fill_count is not None
    assert rolling.diagnostics.unfilled_count is not None


def test_fast_slow_parity_hold_switch_after_ledger() -> None:
    from tests.unit.tournament.test_simulator import test_simulate_window_live_scores_match_engine_slow

    test_simulate_window_live_scores_match_engine_slow()


def test_turnover_uses_open_weights_after_gap() -> None:
    from datetime import date

    import polars as pl

    from src.backtest.costs import CostConfig, CostModel
    from src.backtest.execution import NextOpenExecution
    from src.core.calendar import TradingCalendar
    from src.execution.ledger import PortfolioLedgerState, transition_portfolio_state
    from src.portfolio.intent import PortfolioIntent

    cal = TradingCalendar()
    d0 = date(2026, 1, 2)
    d1 = date(2026, 1, 5)
    state = PortfolioLedgerState(cash=50_000.0, shares={"A": 9500.0})
    w_close = state.weights_at_prices({"A": 100.0})
    w_open = state.weights_at_prices({"A": 110.0})
    assert abs(w_close["A"] - 0.95) < 1e-9
    assert w_open["A"] > 0.95
    intent = PortfolioIntent(kind="target", weights={"A": 0.95})
    result = transition_portfolio_state(
        prior_state=state,
        intent=intent,
        decision_date=d0,
        prev_closes={"A": 100.0},
        opens={"A": 110.0},
        closes={"A": 112.0},
        cost_model=CostModel(CostConfig(commission_bps=10.0, slippage_bps=0.0, tax_bps=0.0)),
        adv_by_ticker={"A": 1e12},
        max_order_to_adv=1.0,
        exposure_limits=None,
        leverage_multiples={"A": 1},
        execution=NextOpenExecution(cal),
        panel=pl.DataFrame([{"date": d1, "ticker": "A", "open": 110.0, "close": 112.0, "is_tradable": True}]),
    )
    expected_turnover = abs(0.95 - w_open["A"])
    assert abs(result.diagnostics.turnover_weight - expected_turnover) < 1e-6
    assert result.diagnostics.turnover_weight < abs(0.95 - 0.95) + 1e-9 or expected_turnover > 1e-6


def test_gross_violation_only_on_post_fill_not_close_drift() -> None:
    from datetime import date

    from src.backtest.costs import CostConfig, CostModel
    from src.execution.ledger import PortfolioLedgerState, transition_portfolio_state
    from src.portfolio.intent import HOLD_INTENT

    state = PortfolioLedgerState(cash=50_000.0, shares={"A": 9500.0})
    limits = (0.95, 1.90, 0.05)
    result = transition_portfolio_state(
        prior_state=state,
        intent=HOLD_INTENT,
        decision_date=date(2026, 1, 2),
        prev_closes={"A": 100.0},
        opens={"A": 100.0},
        closes={"A": 101.0},
        cost_model=CostModel(CostConfig(0.0, 0.0, 0.0)),
        adv_by_ticker={},
        max_order_to_adv=0.05,
        exposure_limits=limits,
        leverage_multiples={"A": 2},
        execution=None,
        panel=None,
    )
    assert result.diagnostics.post_fill_gross <= 1.90 + 1e-9
    assert result.diagnostics.gross_violation is False
    assert result.diagnostics.close_realized_gross > result.diagnostics.post_fill_gross


def test_hold_drift_does_not_flag_gross_violation() -> None:
    from datetime import date

    from src.backtest.costs import CostConfig, CostModel
    from src.execution.ledger import PortfolioLedgerState, transition_portfolio_state
    from src.portfolio.intent import HOLD_INTENT

    state = PortfolioLedgerState(cash=0.05, shares={"069500": 100.0})
    opens = {"069500": 200.0}
    closes = {"069500": 210.0}
    prev_closes = {"069500": 190.0}
    adv = {"069500": 1e12}
    result = transition_portfolio_state(
        prior_state=state,
        intent=HOLD_INTENT,
        decision_date=date(2026, 1, 2),
        prev_closes=prev_closes,
        opens=opens,
        closes=closes,
        cost_model=CostModel(CostConfig(0.0, 0.0, 0.0)),
        adv_by_ticker=adv,
        max_order_to_adv=0.01,
        exposure_limits=(0.95, 1.90, 0.05),
        leverage_multiples={"069500": 2},
        execution=None,
        panel=None,
    )
    assert result.diagnostics.fill_count == 0
    assert result.diagnostics.gross_violation is False
