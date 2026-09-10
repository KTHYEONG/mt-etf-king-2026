from __future__ import annotations

import contextlib
import json
from datetime import date

import polars as pl

from src.alpha.base import AlphaModel, DecisionContext
from src.backtest.costs import CostConfig
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.core.trace import InMemoryTraceSink
from tests.unit.backtest.conftest import build_engine, panel_row


class BuyAndHoldBaseline:
    name = "BuyAndHold"

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]:
        # score 069500 highest
        if "ticker" in snapshot.columns and "mom_20" in snapshot.columns:
            scores = {}
            for row in snapshot.iter_rows(named=True):
                t = str(row.get("ticker"))
                try:
                    scores[t] = float(row.get("mom_20", 0.0))
                except Exception:
                    scores[t] = 0.0
            return scores
        return {"069500": 1.0}


def _make_panel_two_sessions():
    from src.core.calendar import TradingCalendar

    cal = TradingCalendar()
    d1 = date(2026, 1, 2)
    d3 = date(2026, 1, 6)
    sessions = cal.sessions(d1, d3)
    rows = []
    for d in sessions:
        rows.append(panel_row(day=d, ticker="069500", close=100.0, mom_20=0.05))
        rows.append(panel_row(day=d, ticker="114800", close=100.0, mom_20=0.01))
    df = pl.DataFrame(rows)
    with contextlib.suppress(Exception):
        df = df.with_columns(pl.col("date").cast(pl.Date))
    return df, sessions


def test_engine_run_identical_with_and_without_trace() -> None:
    df, sessions = _make_panel_two_sessions()
    engine, cal, filt = build_engine(df, warmup_sessions=1, max_order_to_adv=1.0)
    from src.portfolio.sizing import SizingScheme

    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig())
    model = BuyAndHoldBaseline()
    result_a = engine.run(model, df, config, trace=None)
    result_b = engine.run(model, df, config, trace=InMemoryTraceSink())
    assert result_a.daily["equity"].to_list() == result_b.daily["equity"].to_list()
    assert result_a.daily["ret"].to_list() == result_b.daily["ret"].to_list()
    assert result_a.trades.height == result_b.trades.height
    if result_a.trades.height > 0:
        assert result_a.trades["ticker"].to_list() == result_b.trades["ticker"].to_list()
        assert result_a.trades["side"].to_list() == result_b.trades["side"].to_list()
        assert result_a.trades["weight_after"].to_list() == result_b.trades["weight_after"].to_list()
    assert result_a.unfilled == result_b.unfilled


def test_engine_emits_rejected_candidates_when_enabled() -> None:
    df, sessions = _make_panel_two_sessions()
    engine, cal, filt = build_engine(df, warmup_sessions=1, max_order_to_adv=1.0)
    from src.portfolio.sizing import SizingScheme

    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig())
    sink = InMemoryTraceSink()
    result = engine.run(BuyAndHoldBaseline(), df, config, trace=sink)
    assert sink.sessions
    for s in sink.sessions:
        assert s.n_universe >= 1
    # candidates should have rejected 114800
    found_rejected = False
    found_selected = False
    for c in sink.candidates:
        if c.ticker == "114800" and c.reject_reason == "TOPK_CUT":
            found_rejected = True
        if c.ticker == "069500" and c.selected and c.reject_reason == "":
            found_selected = True
    assert found_rejected
    assert found_selected
    # null sink path does not require emit
    null_res = engine.run(BuyAndHoldBaseline(), df, config, trace=None)
    assert null_res is not None


def test_engine_lazy_when_sink_disabled(monkeypatch) -> None:
    from src.portfolio.selection import explain_selection_drops as orig

    df, sessions = _make_panel_two_sessions()
    engine, cal, filt = build_engine(df, warmup_sessions=1, max_order_to_adv=1.0)
    from src.portfolio.sizing import SizingScheme

    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig())

    def raising(*a, **kw):
        raise AssertionError("should not be called when disabled")

    monkeypatch.setattr("src.backtest.engine_session.explain_selection_drops", raising)
    # disabled should not raise
    engine.run(BuyAndHoldBaseline(), df, config, trace=None)
    # restore and test enabled may call
    monkeypatch.setattr("src.backtest.engine_session.explain_selection_drops", orig)
    sink = InMemoryTraceSink()
    engine.run(BuyAndHoldBaseline(), df, config, trace=sink)


def test_engine_score_exception_emits_gate_fail_closed() -> None:
    class BoomModel:
        name = "boom"

        def score(self, snapshot, context):
            raise RuntimeError("boom")

    df, sessions = _make_panel_two_sessions()
    engine, cal, filt = build_engine(df, warmup_sessions=1, max_order_to_adv=1.0)
    from src.portfolio.sizing import SizingScheme

    config = BacktestConfig(start=sessions[0], end=sessions[-1], capital=1_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig())
    sink = InMemoryTraceSink()
    result = engine.run(BoomModel(), df, config, trace=sink)
    # should not propagate
    assert result is not None
    gates = [g for g in sink.gates if g.gate == "SCORE_EXCEPTION"]
    assert len(gates) >= 1
    assert gates[0].exc_type == "RuntimeError"
    # boom message not in json dumps
    dumped = json.dumps([{"exc_type": g.exc_type, "gate": g.gate} for g in sink.gates])
    assert "boom" not in dumped


from unittest.mock import patch


from src.backtest.engine_session import emit_session_trace
from src.universe.provider import UniverseFilters, UniverseMode, UniverseSnapshot


def test_emit_session_trace_diagnostics_single_snapshot_pass() -> None:
    decision_date = date(2026, 1, 5)
    # Given: a snapshot much wider than the candidate list (STRUCTURAL-like shape)
    tickers = [f"T{i:04d}" for i in range(300)]
    snapshot = pl.DataFrame(
        {
            "date": [decision_date] * len(tickers),
            "ticker": tickers,
            "mom_20": [0.01 * i for i in range(len(tickers))],
            "mom_60": [0.02 * i for i in range(len(tickers))],
        }
    )
    scores = {t: float(len(tickers) - i) for i, t in enumerate(tickers[:50])}
    target = {tickers[0]: 0.95}
    filters = UniverseFilters(mode=UniverseMode.STRUCTURAL)
    snap_universe = UniverseSnapshot(as_of=decision_date, mode=UniverseMode.STRUCTURAL,
                                     tickers=tuple(tickers), dropped={}, filters=filters)
    sink = InMemoryTraceSink()

    class _Universe:
        master = None

    original_filter = pl.DataFrame.filter
    original_getcol = pl.DataFrame.get_column

    # When: the session trace is emitted
    with (
        patch.object(pl.DataFrame, "filter", autospec=True, side_effect=original_filter) as filter_spy,
        patch.object(pl.DataFrame, "get_column", autospec=True, side_effect=original_getcol) as getcol_spy,
    ):
        emit_session_trace(
            sink=sink,
            decision_date=decision_date,
            snap_universe=snap_universe,
            scores=scores,
            target=target,
            target_before_adv=dict(target),
            raw_weights=dict(target),
            fills=[],
            unfilled=[],
            new_weights={},
            current_weights={},
            snapshot=snapshot,
            used_allocate_path=False,
            portfolio_vehicles=None,
            model=object(),
            universe=_Universe(),
            regime_snap=None,
            equity=1_000_000_000.0,
        )

    # Then: diagnostics are still attached, with the snapshot's values
    candidates = list(sink.candidates)
    assert len(candidates) == len(scores)
    by_ticker = {c.ticker: c for c in candidates}
    first = by_ticker[tickers[0]]
    assert first.diagnostics is not None
    assert first.diagnostics["mom_20"] == 0.0
    assert first.diagnostics["mom_60"] == 0.0
    second = by_ticker[tickers[1]]
    assert second.diagnostics is not None
    assert abs(second.diagnostics["mom_20"] - 0.01) < 1e-12
    assert abs(second.diagnostics["mom_60"] - 0.02) < 1e-12
    # And: the snapshot was not rescanned per candidate
    assert filter_spy.call_count == 0
    assert getcol_spy.call_count == 0







def test_emit_session_trace_diagnostics_none_without_ticker_column() -> None:
    decision_date = date(2026, 1, 5)
    filters = UniverseFilters(mode=UniverseMode.DEPLOYMENT)
    snap_universe = UniverseSnapshot(as_of=decision_date, mode=UniverseMode.DEPLOYMENT,
                                     tickers=("A", "B"), dropped={}, filters=filters)

    class _Universe:
        master = None

    def _emit(snapshot: pl.DataFrame) -> list:
        sink = InMemoryTraceSink()
        emit_session_trace(
            sink=sink,
            decision_date=decision_date,
            snap_universe=snap_universe,
            scores={"A": 2.0, "B": 1.0},
            target={"A": 0.95},
            target_before_adv={"A": 0.95},
            raw_weights={"A": 0.95},
            fills=[],
            unfilled=[],
            new_weights={},
            current_weights={},
            snapshot=snapshot,
            used_allocate_path=False,
            portfolio_vehicles=None,
            model=object(),
            universe=_Universe(),
            regime_snap=None,
            equity=1_000_000_000.0,
        )
        return list(sink.candidates)

    # When: the snapshot has no 'ticker' column at all
    no_ticker = _emit(pl.DataFrame({"mom_20": [0.1, 0.2]}))
    # And: when it has tickers but no diagnostic feature columns
    no_diag_cols = _emit(pl.DataFrame({"ticker": ["A", "B"], "close": [100.0, 200.0]}))
    # And: when the diagnostic value is null
    null_diag = _emit(pl.DataFrame({"ticker": ["A", "B"], "mom_20": [None, None]}))

    # Then: diagnostics fail closed to None instead of raising
    assert no_ticker and all(c.diagnostics is None for c in no_ticker)
    assert no_diag_cols and all(c.diagnostics is None for c in no_diag_cols)
    assert null_diag and all(c.diagnostics is None for c in null_diag)



def test_emit_session_trace_diagnostics_skips_non_numeric_values() -> None:
    """A non-numeric diagnostic value is skipped, not fatal (narrow TypeError/ValueError guard)."""
    decision_date = date(2026, 1, 5)
    # Given: mom_20 holds an unconvertible string while mom_60 is a usable float
    snapshot = pl.DataFrame(
        {
            "ticker": ["A", "B"],
            "mom_20": ["not-a-number", "also-bad"],
            "mom_60": [0.5, 0.25],
        }
    )
    filters = UniverseFilters(mode=UniverseMode.DEPLOYMENT)
    snap_universe = UniverseSnapshot(
        as_of=decision_date, mode=UniverseMode.DEPLOYMENT, tickers=("A", "B"), dropped={}, filters=filters
    )
    sink = InMemoryTraceSink()

    class _Universe:
        master = None

    # When
    emit_session_trace(
        sink=sink,
        decision_date=decision_date,
        snap_universe=snap_universe,
        scores={"A": 2.0, "B": 1.0},
        target={"A": 0.95},
        target_before_adv={"A": 0.95},
        raw_weights={"A": 0.95},
        fills=[],
        unfilled=[],
        new_weights={},
        current_weights={},
        snapshot=snapshot,
        used_allocate_path=False,
        portfolio_vehicles=None,
        model=object(),
        universe=_Universe(),
        regime_snap=None,
        equity=1_000_000_000.0,
    )

    # Then: the unconvertible column is dropped while the numeric one survives
    by_ticker = {c.ticker: c for c in sink.candidates}
    assert by_ticker["A"].diagnostics == {"mom_60": 0.5}
    assert by_ticker["B"].diagnostics == {"mom_60": 0.25}
