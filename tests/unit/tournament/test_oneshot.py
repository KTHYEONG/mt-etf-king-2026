def test_oneshot_anchor_starts_sep21_and_horizon_fit() -> None:
    from datetime import date, timedelta

    from src.tournament.distribution import oneshot_anchor_starts

    y2024 = [date(2024, 9, 20)] + [date(2024, 9, 23) + timedelta(days=i) for i in range(40)]
    y2025_short = [date(2025, 9, 22) + timedelta(days=i) for i in range(10)]
    sessions = y2024 + y2025_short
    starts = oneshot_anchor_starts(sessions, month=9, day=21, horizon=36)
    assert starts == (date(2024, 9, 23),)
    assert oneshot_anchor_starts([], month=9, day=21, horizon=36) == ()
    assert oneshot_anchor_starts(y2024, month=9, day=21, horizon=0) == ()


def test_oneshot_window_returns_compounds_and_skips_short() -> None:
    from datetime import date

    from src.tournament.distribution import oneshot_window_returns

    sessions = [date(2024, 9, 23), date(2024, 9, 24), date(2024, 9, 25)]
    daily = [0.10, 0.10, 0.10]
    rows = oneshot_window_returns(daily, sessions, (date(2024, 9, 23),), horizon=2)
    assert rows == ((2024, date(2024, 9, 23), 0.21),)
    assert oneshot_window_returns(daily, sessions, (date(2024, 9, 23),), horizon=4) == ()
    assert oneshot_window_returns([0.1], sessions, (date(2024, 9, 23),), horizon=2) == ()
    assert oneshot_window_returns(daily, sessions, (date(2024, 9, 30),), horizon=2) == ()


def test_oneshot_independent_window_returns_matches_fresh_engine() -> None:
    from datetime import date

    import polars as pl

    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig
    from src.backtest.metrics import compound_returns
    from src.core.calendar import TradingCalendar
    from src.portfolio.sizing import SizingScheme
    from src.tournament.simulator import oneshot_independent_window_returns
    from tests.unit.backtest.conftest import build_engine, panel_row

    class _HoldSwitchModel:
        name = "hold_switch"
        path_dependent = True
        scores_path_independent = False

        def __init__(self) -> None:
            self._held: str | None = None
            self._hold_len: int = 0

        def reset_trackers(self) -> None:
            self._held = None
            self._hold_len = 0

        def score(self, snapshot: object, context: object) -> dict[str, float]:
            tickers: list[str] = []
            try:
                if snapshot is not None and "ticker" in snapshot.columns:  # type: ignore[union-attr]
                    tickers = [str(t) for t in snapshot.get_column("ticker").to_list()]  # type: ignore[union-attr]
            except Exception:
                tickers = []
            uniq = sorted(set(tickers))
            if not uniq:
                return {}
            first, last = uniq[0], uniq[-1]
            if self._held is None or self._held not in uniq:
                self._held = first
                self._hold_len = 1
                return {self._held: 1.0}
            self._hold_len += 1
            if self._hold_len >= 3 and last != first:
                self._held = last
                self._hold_len = 1
            return {str(self._held): 1.0}

    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 2, 10))
    rows: list[dict[str, object]] = []
    for i, d in enumerate(sessions):
        rows.append(panel_row(day=d, ticker="069500", close=30000.0 + i * 50.0, mom_20=0.10, name="KODEX 200", theme="KOSPI"))
        rows.append(panel_row(day=d, ticker="122630", close=20000.0 + i * 80.0, mom_20=0.20, name="KODEX 레버리지", theme="LEV"))
    panel = pl.DataFrame(rows)
    engine, cal2, filt = build_engine(panel, warmup_sessions=1)
    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    horizon = 5
    starts = (sessions[0], sessions[6], sessions[10])
    model = _HoldSwitchModel()
    assert oneshot_independent_window_returns(engine, model, panel, config, (), horizon, cal2) == ()
    assert oneshot_independent_window_returns(engine, model, panel, config, starts, 0, cal2) == ()
    rows_out = oneshot_independent_window_returns(engine, model, panel, config, starts, horizon, cal2)
    assert len(rows_out) == 3
    for year, start, ret in rows_out:
        assert int(year) == int(start.year)
        idx = sessions.index(start)
        end_date = sessions[idx + horizon - 1]
        win_config = BacktestConfig(
            start=start,
            end=end_date,
            capital=config.capital,
            scheme=config.scheme,
            k=config.k,
            filters=config.filters,
            costs=config.costs,
        )
        fresh = _HoldSwitchModel()
        res = engine.run(fresh, panel, win_config)
        daily = res.daily
        ret_col = "ret" if "ret" in daily.columns else "return"
        rets = [float(row.get(ret_col) or 0.0) for row in daily.iter_rows(named=True)]
        independent = compound_returns(rets) if rets else 0.0
        assert abs(float(ret) - float(independent)) < 1e-10


def test_oneshot_independent_window_returns_uses_cache_not_engine() -> None:
    from datetime import date
    from unittest.mock import MagicMock

    import polars as pl

    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig
    from src.core.calendar import TradingCalendar
    from src.portfolio.sizing import SizingScheme
    from src.tournament.simulator import oneshot_independent_window_returns
    from tests.unit.backtest.conftest import build_engine, panel_row

    class _HoldSwitchModel:
        name = "hold_switch"
        path_dependent = True
        scores_path_independent = False

        def __init__(self) -> None:
            self._held: str | None = None
            self._hold_len: int = 0

        def reset_trackers(self) -> None:
            self._held = None
            self._hold_len = 0

        def score(self, snapshot: object, context: object) -> dict[str, float]:
            tickers: list[str] = []
            try:
                if snapshot is not None and "ticker" in snapshot.columns:  # type: ignore[union-attr]
                    tickers = [str(t) for t in snapshot.get_column("ticker").to_list()]  # type: ignore[union-attr]
            except Exception:
                tickers = []
            uniq = sorted(set(tickers))
            if not uniq:
                return {}
            first, last = uniq[0], uniq[-1]
            if self._held is None or self._held not in uniq:
                self._held = first
                self._hold_len = 1
                return {self._held: 1.0}
            self._hold_len += 1
            if self._hold_len >= 3 and last != first:
                self._held = last
                self._hold_len = 1
            return {str(self._held): 1.0}

    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 3, 15))
    panel = pl.DataFrame(
        [panel_row(day=d, ticker="069500", close=30000.0 + i, mom_20=0.2, name="KODEX 200") for i, d in enumerate(sessions)]
    )
    engine, cal2, filt = build_engine(panel)
    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    model = _HoldSwitchModel()
    mock_engine = MagicMock(wraps=engine)
    mock_engine.run = MagicMock(wraps=engine.run)
    mock_engine.execution = engine.execution
    mock_engine.calendar = engine.calendar
    mock_engine.universe = engine.universe
    mock_engine.features = engine.features
    mock_engine.regimes = engine.regimes
    mock_engine._leverage_multiples = engine._leverage_multiples  # type: ignore[method-assign]
    mock_engine._portfolio_exposure_limits = engine._portfolio_exposure_limits  # type: ignore[method-assign]
    starts = (sessions[0], sessions[10])
    rows = oneshot_independent_window_returns(mock_engine, model, panel, config, starts, 5, cal2)
    assert len(rows) == 2
    assert mock_engine.run.call_count == 0

