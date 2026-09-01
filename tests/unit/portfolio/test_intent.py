def test_cash_intent_liquidates_positions() -> None:
    from datetime import date

    import polars as pl

    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig
    from src.core.calendar import TradingCalendar
    from src.portfolio.intent import CASH_INTENT
    from src.portfolio.sizing import SizingScheme
    from tests.unit.backtest.conftest import build_engine

    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 8, 14), date(2026, 8, 21))
    panel = pl.DataFrame(
        [
            {"date": d, "ticker": "A", "close": 100.0, "open": 100.0, "is_tradable": True}
            for d in sessions
        ]
    )
    engine, _, filt = build_engine(panel)

    class _CashModel:
        name = "cash_model"

        def score(self, snapshot: object, context: object) -> dict[str, float]:
            return {"A": 1.0}

        def allocate(self, scores: dict[str, float], **kwargs: object) -> object:
            return CASH_INTENT

    config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0.0, 0.0, 0.0),
    )
    result = engine.run(_CashModel(), panel, config)
    assert result.trades.height >= 1
    last_trade = result.trades.sort("decision_date").row(-1, named=True)
    assert float(last_trade.get("weight_after", 1.0)) == 0.0


def test_empty_scores_hold_not_cash() -> None:
    from src.portfolio.intent import HOLD_INTENT, resolve_portfolio_intent

    intent = resolve_portfolio_intent({}, current_weights={"A": 0.5}, score_failed=True)
    assert intent.kind == HOLD_INTENT.kind
    assert intent.weights == {}
