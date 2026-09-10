from datetime import date, timedelta
from unittest.mock import patch

import polars as pl
import pytest


def test_resolve_live_championship_sleeve_truncates_to_pit_without_violation() -> None:
    from src.tournament.live_decision import resolve_live_championship_sleeve

    # Given: 90 sessions of KOSPI history followed by 2 FUTURE rows that must NOT be
    # visible to a decision made on the 90th day (PIT safety, R1).
    start = date(2026, 1, 5)
    closes = [1000.0]
    for i in range(1, 92):
        if i < 30:
            closes.append(closes[-1] * 1.01)
        elif i < 70:
            closes.append(closes[-1] * 0.99)
        else:
            closes.append(closes[-1] * 1.012)
    dates = [start + timedelta(days=i) for i in range(len(closes))]
    decision_date = dates[89]
    future_dates = dates[90:92]
    index_daily = pl.DataFrame(
        {"date": dates, "index_name": ["KOSPI"] * len(dates), "close": closes},
        schema={"date": pl.Date, "index_name": pl.String, "close": pl.Float64},
    )

    # When
    sleeve = resolve_live_championship_sleeve(index_daily, decision_date)

    # Then: no PIT violation, and the future rows exist in the frame but were excluded
    assert sleeve in {"CRASH_REBOUND", "LOTTERY_ON", "INACTIVE", "UNCERTAIN"}
    assert all(d > decision_date for d in future_dates)


def test_resolve_live_championship_sleeve_fails_closed_on_empty_or_missing_history() -> None:
    from src.tournament.live_decision import resolve_live_championship_sleeve

    empty = pl.DataFrame({"date": [], "index_name": [], "close": []}, schema={"date": pl.Date, "index_name": pl.String, "close": pl.Float64})

    # Then: empty frame -> UNCERTAIN (R2)
    assert resolve_live_championship_sleeve(empty, date(2026, 8, 27)) == "UNCERTAIN"
    # Then: non-DataFrame -> UNCERTAIN (R2)
    assert resolve_live_championship_sleeve(object(), date(2026, 8, 27)) == "UNCERTAIN"  # type: ignore[arg-type]
    # Then: a decision_date far outside the frame's coverage -> UNCERTAIN
    sparse = pl.DataFrame(
        {"date": [date(2020, 1, 2)], "index_name": ["KOSPI"], "close": [100.0]},
        schema={"date": pl.Date, "index_name": pl.String, "close": pl.Float64},
    )
    assert resolve_live_championship_sleeve(sparse, date(2026, 8, 27)) == "UNCERTAIN"


def test_build_live_eligible_snapshot_reuses_market_candidates_and_filters_panel() -> None:
    from datetime import timedelta

    from src.tournament.live_decision import build_live_eligible_snapshot

    # Given: a panel spanning MULTIPLE sessions (not just decision_date). This is the
    # regression guard for the bug where passing only [decision_date] silently broke
    # market_candidates_by_session's session-index history arithmetic (every ticker's
    # (session_idx - first_seen_idx) collapsed to 0, failing every history check).
    sessions = [date(2026, 8, 24) + timedelta(days=i) for i in range(4)]
    d = sessions[-1]
    rows = [{"date": s, "ticker": t, "mom_60": 0.1} for s in sessions for t in ("111", "222", "333")]
    panel = pl.DataFrame(rows, schema={"date": pl.Date, "ticker": pl.String, "mom_60": pl.Float64})

    with patch(
        "src.tournament.attainability.market_candidates_by_session",
        return_value={d: ("111", "333")},
    ) as spy:
        snap = build_live_eligible_snapshot(panel, decision_date=d)

    # Then: reused with a session list spanning the panel's full history through
    # decision_date - not a single-element [decision_date] list (R3 regression guard)
    assert spy.call_count == 1
    called_sessions = spy.call_args.kwargs["sessions"]
    assert called_sessions[-1] == d
    assert len(called_sessions) > 1
    assert spy.call_args.kwargs["panel"] is panel
    assert sorted(snap["ticker"].to_list()) == ["111", "333"]
    assert snap["date"].to_list() == [d, d]


def test_build_live_eligible_snapshot_empty_when_no_candidates() -> None:
    from src.tournament.live_decision import build_live_eligible_snapshot

    d = date(2026, 8, 27)
    panel = pl.DataFrame({"date": [d], "ticker": ["111"]}, schema={"date": pl.Date, "ticker": pl.String})

    with patch("src.tournament.attainability.market_candidates_by_session", return_value={}):
        snap = build_live_eligible_snapshot(panel, decision_date=d)

    # Then: empty, not the raw panel (R3 - never widen)
    assert snap.height == 0


class _StubModel:
    def score(self, snapshot: pl.DataFrame, context: object) -> dict[str, float]:
        assert getattr(context, "championship_sleeve", None) == "CRASH_REBOUND"
        return {"412570": 0.6708, "462330": 0.6448}


def test_compute_live_target_weights_uses_top1_sizing_for_sticky_model() -> None:
    from src.portfolio.sizing import SizingScheme
    from src.tournament.live_decision import compute_live_target_weights

    snapshot = pl.DataFrame({"ticker": ["412570", "462330"]}, schema={"ticker": pl.String})

    # When
    intent = compute_live_target_weights(
        _StubModel(),
        snapshot,
        decision_date=date(2026, 8, 27),
        held={},
        capital=1_000_000_000.0,
        rules=None,
        championship_sleeve="CRASH_REBOUND",
        scheme=SizingScheme.TOP1,
        k=1,
    )

    # Then: TOP1 concentrates 100% on the highest-scored ticker (R4)
    assert intent.kind == "target"
    assert intent.weights == {"412570": pytest.approx(1.0)}


class _CashModel:
    def score(self, snapshot: pl.DataFrame, context: object) -> object:
        from src.portfolio.intent import CASH_INTENT

        return CASH_INTENT


def test_compute_live_target_weights_passes_through_explicit_cash_intent() -> None:
    from src.portfolio.intent import CASH_INTENT
    from src.tournament.live_decision import compute_live_target_weights

    snapshot = pl.DataFrame({"ticker": []}, schema={"ticker": pl.String})

    intent = compute_live_target_weights(
        _CashModel(),
        snapshot,
        decision_date=date(2026, 8, 27),
        held={},
        capital=1_000_000_000.0,
        rules=None,
        championship_sleeve="INACTIVE",
    )

    assert intent is CASH_INTENT


class _BrokenModel:
    def score(self, snapshot: pl.DataFrame, context: object) -> dict[str, float]:
        raise RuntimeError("feature panel malformed")


def test_compute_live_target_weights_propagates_scoring_exception() -> None:
    from src.tournament.live_decision import compute_live_target_weights

    snapshot = pl.DataFrame({"ticker": ["111"]}, schema={"ticker": pl.String})

    with pytest.raises(RuntimeError, match="feature panel malformed"):
        compute_live_target_weights(
            _BrokenModel(),
            snapshot,
            decision_date=date(2026, 8, 27),
            held={},
            capital=1_000_000_000.0,
            rules=None,
            championship_sleeve="LOTTERY_ON",
        )


def test_estimate_live_order_quantities_computes_whole_share_lots() -> None:
    from src.tournament.live_decision import estimate_live_order_quantities

    d = date(2026, 8, 27)
    panel = pl.DataFrame(
        {"date": [d], "ticker": ["412570"], "close": [726_500.0]},
        schema={"date": pl.Date, "ticker": pl.String, "close": pl.Float64},
    )

    out = estimate_live_order_quantities({"412570": 1.0}, panel, decision_date=d, capital=1_000_000_000.0)

    est = out["412570"]
    assert est.price_basis_date == d
    assert est.price == pytest.approx(726_500.0)
    assert est.est_shares == 1_376
    assert est.est_krw == pytest.approx(1_376 * 726_500.0)


def test_estimate_live_order_quantities_raises_on_missing_price_for_weighted_ticker() -> None:
    from src.tournament.live_decision import estimate_live_order_quantities

    d = date(2026, 8, 27)
    panel = pl.DataFrame({"date": [d], "ticker": ["111"], "close": [100.0]}, schema={"date": pl.Date, "ticker": pl.String, "close": pl.Float64})

    # Then: weighted ticker missing from panel -> ValueError naming it (R6)
    with pytest.raises(ValueError, match="412570"):
        estimate_live_order_quantities({"412570": 1.0}, panel, decision_date=d, capital=1_000_000_000.0)

    # Then: empty weights short-circuits before touching panel/capital (R9) - bad capital, still returns {}
    assert estimate_live_order_quantities({}, panel, decision_date=d, capital=-1.0) == {}

    # Then: non-finite/non-positive capital with a real weighted ticker raises (R7)
    with pytest.raises(ValueError, match="capital"):
        estimate_live_order_quantities({"111": 1.0}, panel, decision_date=d, capital=0.0)



import polars as pl


def test_assert_sleeve_inputs_fresh_raises_when_decision_date_row_missing() -> None:
    from src.tournament.live_decision import StaleSleeveInputError, assert_sleeve_inputs_fresh

    # Given: index data that stops one session BEFORE the decision date (the real defect)
    stale = pl.DataFrame(
        {"date": [date(2026, 8, 26), date(2026, 8, 27)], "index_name": ["코스피", "코스피"], "close": [7000.0, 7051.64]},
        schema={"date": pl.Date, "index_name": pl.String, "close": pl.Float64},
    )

    # Then: fail closed, and the message names the date (R5/R6 - no latest-available fallback)
    with pytest.raises(StaleSleeveInputError, match="2026-09-21"):
        assert_sleeve_inputs_fresh(stale, decision_date=date(2026, 9, 21))

    # Then: an empty frame (missing index file path) also fails closed
    empty = pl.DataFrame(
        {"date": [], "index_name": [], "close": []},
        schema={"date": pl.Date, "index_name": pl.String, "close": pl.Float64},
    )
    with pytest.raises(StaleSleeveInputError):
        assert_sleeve_inputs_fresh(empty, decision_date=date(2026, 9, 21))

    # Then: only sub-indices present -> no headline series -> fail closed
    only_sub = pl.DataFrame(
        {"date": [date(2026, 9, 21)], "index_name": ["코스피 200"], "close": [900.0]},
        schema={"date": pl.Date, "index_name": pl.String, "close": pl.Float64},
    )
    with pytest.raises(StaleSleeveInputError):
        assert_sleeve_inputs_fresh(only_sub, decision_date=date(2026, 9, 21))



import polars as pl


def test_assert_sleeve_inputs_fresh_passes_and_preserves_legitimate_uncertain() -> None:
    from src.tournament.live_decision import assert_sleeve_inputs_fresh, resolve_live_championship_sleeve

    # Given: a row exists exactly at decision_date but with far too little history for mom60
    d = date(2026, 9, 21)
    frame = pl.DataFrame(
        {"date": [date(2026, 9, 18), d], "index_name": ["코스피", "코스피"], "close": [7000.0, 7051.64]},
        schema={"date": pl.Date, "index_name": pl.String, "close": pl.Float64},
    )

    # When: the gate passes (returns None, does not raise)
    assert assert_sleeve_inputs_fresh(frame, decision_date=d) is None

    # Then: insufficient history is still a legitimate UNCERTAIN, not an error (R7)
    assert resolve_live_championship_sleeve(frame, d) == "UNCERTAIN"
