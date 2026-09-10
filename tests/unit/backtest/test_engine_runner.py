"""Facade fidelity for the src backtest engine runner split (P5)."""

# ruff: noqa: C420 -- contract skeleton uses dict comprehension verbatim
from __future__ import annotations


def test_engine_runner_canonical_home() -> None:
    import src.backtest.engine as facade
    from src.backtest.engine_runner import BacktestEngine

    assert facade.BacktestEngine is BacktestEngine
    assert BacktestEngine.__module__ == "src.backtest.engine_runner"


def test_engine_run_injects_championship_sleeve_into_every_decision_context() -> None:
    from datetime import date

    import polars as pl

    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig
    from src.core.calendar import TradingCalendar
    from src.portfolio.sizing import SizingScheme
    from tests.unit.backtest.conftest import build_engine, panel_row

    class _SleeveRecorder:
        name = "recorder"
        path_dependent = False
        scores_path_independent = True

        def __init__(self) -> None:
            self.seen: dict[date, str | None] = {}

        def score(self, snapshot: pl.DataFrame, context: object) -> dict[str, float]:
            self.seen[context.decision_date] = context.championship_sleeve  # type: ignore[attr-defined]
            return {}

    # Given: a panel whose first session precedes config.start so the pre-start
    # bootstrap path (engine_session) also builds a DecisionContext
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 1, 2), date(2026, 1, 15))
    panel = pl.DataFrame([panel_row(day=d, ticker="069500", close=100.0) for d in sessions])
    engine, _, filt = build_engine(panel, max_order_to_adv=1.0)
    engine.championship_sleeves = {d: "LOTTERY_ON" for d in sessions}
    config = BacktestConfig(
        start=sessions[1],
        end=sessions[-1],
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(0, 0, 0),
    )
    model = _SleeveRecorder()

    # When
    engine.run(model, panel, config)

    # Then: every scored session, including the pre-start bootstrap, saw the sleeve
    assert sessions[0] in model.seen, "pre-start bootstrap context must be exercised"
    assert set(model.seen) >= set(sessions[1:])
    assert set(model.seen.values()) == {"LOTTERY_ON"}

    # And: a decision_date absent from the mapping resolves to None, not an error
    engine2, _, filt2 = build_engine(panel, max_order_to_adv=1.0)
    engine2.championship_sleeves = {}
    model2 = _SleeveRecorder()
    engine2.run(model2, panel, config)
    assert set(model2.seen.values()) == {None}


def test_backtest_engine_declares_championship_sleeves_default_none() -> None:
    from datetime import date

    import polars as pl

    from tests.unit.backtest.conftest import build_engine, panel_row

    cal_rows = [panel_row(day=date(2026, 1, 2), ticker="069500", close=100.0)]
    engine, _, _ = build_engine(pl.DataFrame(cal_rows))

    assert hasattr(engine, "championship_sleeves")
    assert engine.championship_sleeves is None
