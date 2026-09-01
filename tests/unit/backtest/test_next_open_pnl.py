def test_next_open_gap_not_credited_to_new_buy() -> None:
    from src.backtest.pnl import compute_next_open_session_return

    overnight, intraday, effective = compute_next_open_session_return(
        weights_before_open={},
        weights_after_open={"A": 1.0},
        prev_closes={"A": 100.0},
        opens={"A": 120.0},
        closes={"A": 120.0},
    )
    assert abs(overnight) < 1e-12
    assert abs(intraday) < 1e-12
    assert abs(effective) < 1e-12


def test_next_open_overnight_on_held_position() -> None:
    from src.backtest.pnl import compute_next_open_session_return

    overnight, intraday, effective = compute_next_open_session_return(
        weights_before_open={"A": 1.0},
        weights_after_open={"A": 1.0},
        prev_closes={"A": 100.0},
        opens={"A": 120.0},
        closes={"A": 120.0},
    )
    assert abs(overnight - 0.2) < 1e-12
    assert abs(intraday) < 1e-12
    assert abs(effective - 0.2) < 1e-12


def test_next_open_engine_matches_helper() -> None:
    from datetime import date

    import polars as pl

    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig
    from src.backtest.pnl import compute_next_open_session_return
    from src.core.calendar import TradingCalendar
    from src.portfolio.sizing import SizingScheme
    from tests.unit.backtest.conftest import build_engine

    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 8, 14), date(2026, 8, 18))
    panel = pl.DataFrame(
        [
            {"date": date(2026, 8, 14), "ticker": "A", "close": 100.0, "open": 100.0, "is_tradable": True},
            {"date": date(2026, 8, 18), "ticker": "A", "close": 120.0, "open": 120.0, "is_tradable": True},
        ]
    )
    engine, _, filt = build_engine(panel)

    class _BuyDay0:
        name = "buy_day0"

        def score(self, snapshot: object, context: object) -> dict[str, float]:
            return {"A": 1.0}

        def allocate(self, scores: dict[str, float], **kwargs: object) -> dict[str, float]:
            return {"A": 1.0}

    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    result = engine.run(_BuyDay0(), panel, config)
    row = result.daily.filter(pl.col("date") == date(2026, 8, 18)).row(0, named=True)
    _, _, expected = compute_next_open_session_return(
        weights_before_open={},
        weights_after_open={"A": 1.0},
        prev_closes={"A": 100.0},
        opens={"A": 120.0},
        closes={"A": 120.0},
    )
    assert abs(float(row["ret"]) - expected) < 1e-9
