from __future__ import annotations


def test_protocol_cell_key_identifies_canonical_grid_slot() -> None:
    from src.backtest.costs import CostConfig
    from src.tournament.eval_cache import is_canonical_protocol_cell, plan_control_evaluations, protocol_cell_key
    from src.tournament.harness import iter_protocol_cases
    canonical = CostConfig(commission_bps=3.0, slippage_bps=5.0, spread_bps=0.0)
    assert protocol_cell_key(canonical, 0.01) == "0300_0500_0010"
    assert is_canonical_protocol_cell(canonical, 0.01) is True
    assert is_canonical_protocol_cell(CostConfig(0.0, 0.0, 0.0), 0.01) is False
    grid = list(iter_protocol_cases("grid"))
    flags = plan_control_evaluations("grid", grid)
    assert len(grid) == 36
    assert flags.count(True) == 1
    true_cases = [c for c, f in zip(grid, flags, strict=False) if f]  # noqa: B905
    assert len(true_cases) == 1
    assert is_canonical_protocol_cell(true_cases[0][0], true_cases[0][1]) is True
    assert plan_control_evaluations("single", list(iter_protocol_cases("single"))) == [True]


def test_control_rolling_cache_runs_factory_once_per_key() -> None:
    from src.tournament.distribution import ReturnDistribution
    from src.tournament.eval_cache import ControlRollingCache
    cache = ControlRollingCache()
    calls = {"n": 0}
    def factory() -> ReturnDistribution:
        calls["n"] += 1
        return ReturnDistribution.summarise(name="B0", returns=[0.1, 0.2], horizon=2, thresholds=[0.30], tail_weights={0.9: 1.0})
    a = cache.get_or_run("k1", factory)
    b = cache.get_or_run("k1", factory)
    assert a is b
    assert calls["n"] == 1
    cache.get_or_run("k2", factory)
    assert calls["n"] == 2
    assert cache.hits == 1
    assert cache.misses == 2


def test_run_rolling_forwards_shared_close_map() -> None:
    from datetime import date
    from unittest.mock import MagicMock
    import polars as pl
    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig
    from src.backtest.session_cache import build_close_map
    from src.core.calendar import TradingCalendar
    from src.portfolio.policy import PortfolioPolicy
    from src.portfolio.sizing import ConfidenceSizingConfig, SizingScheme
    from src.tournament.eval_mode import resolve_eval_flags
    from src.tournament.simulator import TournamentSimulator
    from tests.unit.backtest.conftest import build_engine, panel_row
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 10))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=30000.0 + i) for i, d in enumerate(sessions)])
    engine, cal2, filt = build_engine(panel)
    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig(0.0, 0.0, 0.0))
    policy = PortfolioPolicy(sizing_config=ConfidenceSizingConfig())
    policy.name = "spy"
    resolve_eval_flags(policy, "adoption")
    def _score(snapshot, ctx):
        return {"069500": 1.0}
    policy.score = _score
    cmap = build_close_map(panel)
    wrapped = MagicMock(wraps=engine.run)
    engine.run = wrapped
    sim = TournamentSimulator(engine, cal2)
    sim.run_rolling(policy, panel, config, horizon=2, path_dependent=False, close_map=cmap)
    assert wrapped.call_count == 1
    kwargs = wrapped.call_args.kwargs
    assert kwargs.get("close_map") is cmap or (wrapped.call_args.args and False)  # noqa: SIM223
    engine.run = wrapped._mock_wraps
    sim2 = TournamentSimulator(engine, cal2)
    a = sim2.run_rolling(policy, panel, config, horizon=2, path_dependent=False, close_map=cmap)
    b = sim2.run_rolling(policy, panel, config, horizon=2, path_dependent=False, close_map=None)
    assert a.returns == b.returns
