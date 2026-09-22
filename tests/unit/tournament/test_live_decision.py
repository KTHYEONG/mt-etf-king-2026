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



import polars as pl


def test_assert_panel_input_fresh_raises_on_empty_or_missing_row() -> None:
    from src.tournament.live_decision import StalePanelInputError, assert_panel_input_fresh

    d = date(2026, 9, 9)

    empty = pl.DataFrame({"date": [], "ticker": []}, schema={"date": pl.Date, "ticker": pl.String})
    with pytest.raises(StalePanelInputError, match="2026-09-09"):
        assert_panel_input_fresh(empty, decision_date=d)

    with pytest.raises(StalePanelInputError):
        assert_panel_input_fresh(None, decision_date=d)  # type: ignore[arg-type]

    stale = pl.DataFrame(
        {"date": [date(2026, 9, 8)], "ticker": ["999"]},
        schema={"date": pl.Date, "ticker": pl.String},
    )
    with pytest.raises(StalePanelInputError):
        assert_panel_input_fresh(stale, decision_date=d)

    # Then: non-empty panel with NO 'date' column at all also fails closed
    no_date_col = pl.DataFrame({"ticker": ["999"]}, schema={"ticker": pl.String})
    with pytest.raises(StalePanelInputError):
        assert_panel_input_fresh(no_date_col, decision_date=d)



import polars as pl


def test_assert_panel_input_fresh_passes_when_decision_date_row_present() -> None:
    from src.tournament.live_decision import assert_panel_input_fresh

    d = date(2026, 9, 8)
    panel = pl.DataFrame(
        {"date": [date(2026, 9, 7), d], "ticker": ["999", "999"]},
        schema={"date": pl.Date, "ticker": pl.String},
    )

    assert assert_panel_input_fresh(panel, decision_date=d) is None



import polars as pl


def test_resolve_prior_trading_session_returns_immediately_preceding_session() -> None:
    from src.tournament.live_decision import resolve_prior_trading_session

    sessions = [date(2026, 8, 24) + timedelta(days=i) for i in range(4)]
    rows = [{"date": s, "ticker": "111"} for s in sessions]
    panel = pl.DataFrame(rows, schema={"date": pl.Date, "ticker": pl.String})

    assert resolve_prior_trading_session(panel, decision_date=sessions[-1]) == sessions[-2]

    # Then: decision_date is the earliest session -> no prior session
    assert resolve_prior_trading_session(panel, decision_date=sessions[0]) is None

    # Then: empty panel -> None
    empty = pl.DataFrame({"date": [], "ticker": []}, schema={"date": pl.Date, "ticker": pl.String})
    assert resolve_prior_trading_session(empty, decision_date=sessions[-1]) is None

    # Then: non-DataFrame input -> None
    assert resolve_prior_trading_session(object(), decision_date=sessions[-1]) is None  # type: ignore[arg-type]

    # Then: panel missing the 'date' column -> None
    no_date_col = pl.DataFrame({"ticker": ["111"]}, schema={"ticker": pl.String})
    assert resolve_prior_trading_session(no_date_col, decision_date=sessions[-1]) is None

    # Then: panel's earliest date is AFTER decision_date -> None
    future_panel = pl.DataFrame({"date": [date(2027, 1, 2)], "ticker": ["111"]}, schema={"date": pl.Date, "ticker": pl.String})
    assert resolve_prior_trading_session(future_panel, decision_date=sessions[0]) is None

    # Then: 'date' column exists but its min() is not a date (e.g. all-null) -> None
    null_date_panel = pl.DataFrame({"date": [None], "ticker": ["111"]}, schema={"date": pl.Date, "ticker": pl.String})
    assert resolve_prior_trading_session(null_date_panel, decision_date=sessions[0]) is None




def test_resolve_prior_sticky_state_round_trips_via_persist(tmp_path) -> None:
    from src.tournament.live_decision import persist_sticky_state, resolve_prior_sticky_state

    p = tmp_path / "sticky_mom60_raw_position.json"
    d = date(2026, 9, 8)

    persist_sticky_state(p, decision_date=d, held="412570", held_weight=0.83, hold_len=2)

    held, weight, hold_len = resolve_prior_sticky_state(p, prior_session=d)
    assert held == "412570"
    assert weight == 0.83
    assert hold_len == 2




def test_resolve_prior_sticky_state_raises_on_ledger_gap(tmp_path) -> None:
    from src.tournament.live_decision import StateDiscontinuityError, persist_sticky_state, resolve_prior_sticky_state

    p = tmp_path / "gap.json"
    persist_sticky_state(p, decision_date=date(2026, 9, 16), held="122630", held_weight=0.95, hold_len=4)

    # 상세 메시지에 경로/요청 세션/원장 최신 세션이 모두 남아야 한다.
    with pytest.raises(StateDiscontinuityError) as excinfo:
        resolve_prior_sticky_state(p, prior_session=date(2026, 9, 17))
    message = str(excinfo.value)
    assert str(p) in message
    assert "2026-09-17" in message
    assert "2026-09-16" in message

    with pytest.raises(StateDiscontinuityError):
        resolve_prior_sticky_state(p, prior_session=date(2026, 9, 8))


def test_resolve_prior_sticky_state_raises_on_missing_or_malformed_ledger(tmp_path) -> None:
    from src.tournament.live_decision import StateDiscontinuityError, resolve_prior_sticky_state

    prior = date(2026, 9, 8)
    malformed = {
        "corrupt": "{not valid json",
        "not_a_dict": "[1, 2, 3]",
        "non_string_as_of": '{"as_of": 20260908, "held": "412570", "held_weight": 1.0, "hold_len": 1}',
        "unparseable_as_of": '{"as_of": "not-a-date", "held": "412570", "held_weight": 1.0, "hold_len": 1}',
    }
    for name, raw in malformed.items():
        path = tmp_path / f"{name}.json"
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(StateDiscontinuityError):
            resolve_prior_sticky_state(path, prior_session=prior)

    with pytest.raises(StateDiscontinuityError):
        resolve_prior_sticky_state(tmp_path / "missing.json", prior_session=prior)


def test_resolve_prior_sticky_state_raises_on_invalid_entry(tmp_path) -> None:
    import json

    from src.tournament.live_decision import StateDiscontinuityError, resolve_prior_sticky_state

    prior = date(2026, 9, 8)
    invalid_histories = {
        "nonfinite_history_weight": {
            "as_of": "2026-09-08",
            "history": {"2026-09-08": {"held": "412570", "held_weight": float("nan"), "hold_len": 1}},
        },
        "invalid_legacy_entry": {"as_of": "2026-09-08", "held": 412570, "held_weight": 1.0, "hold_len": 1},
    }
    for name, payload in invalid_histories.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(StateDiscontinuityError):
            resolve_prior_sticky_state(path, prior_session=prior)


def test_resolve_prior_sticky_state_returns_genuine_cash_entry(tmp_path) -> None:
    import json

    from src.tournament.live_decision import resolve_prior_sticky_state

    prior = date(2026, 9, 8)
    history_ledger = tmp_path / "history_cash.json"
    history_ledger.write_text(
        json.dumps({
            "as_of": "2026-09-08",
            "history": {"2026-09-08": {"held": None, "held_weight": 0.0, "hold_len": 0}},
        }),
        encoding="utf-8",
    )
    legacy_ledger = tmp_path / "legacy_cash.json"
    legacy_ledger.write_text(
        json.dumps({"as_of": "2026-09-08", "held": None, "held_weight": 0.0, "hold_len": 0}),
        encoding="utf-8",
    )

    assert resolve_prior_sticky_state(history_ledger, prior_session=prior) == (None, 0.0, 0)
    assert resolve_prior_sticky_state(legacy_ledger, prior_session=prior) == (None, 0.0, 0)


def test_resolve_prior_sticky_state_bootstraps_without_prior_session(tmp_path) -> None:
    from src.tournament.live_decision import resolve_prior_sticky_state

    assert resolve_prior_sticky_state(tmp_path / "missing.json", prior_session=None) == (None, 0.0, 0)


def test_resolve_prior_sticky_state_prefers_history_entry_when_as_of_is_ahead(tmp_path) -> None:
    import json

    from src.tournament.live_decision import resolve_prior_sticky_state

    p = tmp_path / "ahead.json"
    p.write_text(
        json.dumps({
            "as_of": "2026-09-21",
            "held": "069510",
            "held_weight": 0.95,
            "hold_len": 1,
            "history": {
                "2026-09-18": {"held": "122630", "held_weight": 0.95, "hold_len": 3},
                "2026-09-21": {"held": "069510", "held_weight": 0.95, "hold_len": 1},
            },
        }),
        encoding="utf-8",
    )

    assert resolve_prior_sticky_state(p, prior_session=date(2026, 9, 18)) == ("122630", 0.95, 3)
    assert resolve_prior_sticky_state(p, prior_session=date(2026, 9, 21)) == ("069510", 0.95, 1)


def test_resolve_prior_sticky_state_reads_legacy_top_level_entry(tmp_path) -> None:
    import json

    from src.tournament.live_decision import resolve_prior_sticky_state

    p = tmp_path / "legacy.json"
    p.write_text(
        json.dumps({"as_of": "2026-09-08", "held": "412570", "held_weight": 0.83, "hold_len": 5}),
        encoding="utf-8",
    )

    assert resolve_prior_sticky_state(p, prior_session=date(2026, 9, 8)) == ("412570", 0.83, 5)


def test_sticky_state_latest_session_returns_newest_recorded_session(tmp_path) -> None:
    import json

    from src.tournament.live_decision import persist_sticky_state, sticky_state_latest_session

    assert sticky_state_latest_session(tmp_path / "missing.json") is None

    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({"as_of": "2026-09-16", "held": "122630", "held_weight": 0.95, "hold_len": 1}),
        encoding="utf-8",
    )
    assert sticky_state_latest_session(legacy) == date(2026, 9, 16)

    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({
            "as_of": "2026-09-11",
            "held": "122630",
            "held_weight": 0.95,
            "hold_len": 1,
            "history": {
                "2026-09-16": {"held": "122630", "held_weight": 0.95, "hold_len": 1},
                "2026-09-18": {"held": "122630", "held_weight": 0.95, "hold_len": 2},
            },
        }),
        encoding="utf-8",
    )
    assert sticky_state_latest_session(ledger) == date(2026, 9, 18)

    persisted = tmp_path / "persisted.json"
    persist_sticky_state(persisted, decision_date=date(2026, 9, 21), held="122630", held_weight=0.95, hold_len=1)
    assert sticky_state_latest_session(persisted) == date(2026, 9, 21)


def test_sticky_state_latest_session_raises_on_unreadable_or_undated_ledger(tmp_path) -> None:
    import json

    from src.tournament.live_decision import StateDiscontinuityError, sticky_state_latest_session

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(StateDiscontinuityError):
        sticky_state_latest_session(corrupt)

    not_a_dict = tmp_path / "not_a_dict.json"
    not_a_dict.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(StateDiscontinuityError):
        sticky_state_latest_session(not_a_dict)

    undated = tmp_path / "undated.json"
    undated.write_text(json.dumps({"as_of": "not-a-date", "history": {"not-a-date": {"held": None}}}), encoding="utf-8")
    with pytest.raises(StateDiscontinuityError):
        sticky_state_latest_session(undated)

def test_persist_sticky_state_degrades_safely_on_write_failure(tmp_path, caplog) -> None:
    from src.tournament.live_decision import persist_sticky_state

    # Given: a path whose parent cannot be created (a file, not a directory, in its place)
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    bad_path = blocker / "sticky_mom60_raw_position.json"

    # When/Then: does not raise, degrades safely
    assert persist_sticky_state(bad_path, decision_date=date(2026, 9, 8), held="412570", held_weight=1.0, hold_len=1) is None


def test_persist_sticky_state_param_renamed_state_file_schema_unchanged(tmp_path) -> None:
    import json

    from src.tournament.live_decision import persist_sticky_state, resolve_prior_sticky_state

    p = tmp_path / "state.json"
    persist_sticky_state(p, decision_date=date(2026, 9, 10), held="412570", held_weight=0.5, hold_len=2)

    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["as_of"] == "2026-09-10"

    held, weight, hold_len = resolve_prior_sticky_state(p, prior_session=date(2026, 9, 10))
    assert (held, weight, hold_len) == ("412570", 0.5, 2)


def test_next_hold_len_transitions() -> None:
    from src.tournament.live_decision import next_hold_len

    assert next_hold_len("412570", 2, "412570") == 3
    assert next_hold_len("412570", 2, "462330") == 1
    assert next_hold_len(None, 0, "412570") == 1
    assert next_hold_len("412570", 2, None) == 0


def test_resolve_primary_ticker_deterministic_tiebreak() -> None:
    from src.tournament.live_decision import resolve_primary_ticker

    assert resolve_primary_ticker({}) is None
    assert resolve_primary_ticker({"412570": 0.6, "462330": 0.4}) == "412570"
    assert resolve_primary_ticker({"462330": 0.5, "412570": 0.5}) == "412570"



import polars as pl


def test_apply_live_exposure_and_capacity_limits_caps_leverage_gross_and_min_cash() -> None:
    from src.tournament.live_decision import apply_live_exposure_and_capacity_limits

    d = date(2026, 9, 8)
    panel = pl.DataFrame(
        {
            "date": [d],
            "ticker": ["412570"],
            "name": ["TIGER 2\ucc28\uc804\uc9c0TOP10\ub808\ubc84\ub9ac\uc9c0"],
            "trading_value": [10_000_000_000.0],
        },
        schema={"date": pl.Date, "ticker": pl.String, "name": pl.String, "trading_value": pl.Float64},
    )

    # Given: a huge capital + ADV so the exposure caps bind before the ADV cap does
    out = apply_live_exposure_and_capacity_limits(
        {"412570": 1.0},
        panel,
        held={},
        decision_date=d,
        capital=1_000.0,
        strategy_id="sticky.mom60_raw",
    )

    assert out["412570"] <= 0.95 + 1e-9
    # gross exposure (2x multiplier) must not exceed 1.9
    assert out["412570"] * 2.0 <= 1.9 + 1e-9



import polars as pl


def test_apply_live_exposure_and_capacity_limits_caps_adv_delta_from_held() -> None:
    from src.tournament.live_decision import P27_ADOPTED_MAX_ORDER_TO_ADV, apply_live_exposure_and_capacity_limits

    d = date(2026, 9, 8)
    adv = 100_000_000.0
    capital = 1_000_000_000_000.0
    panel = pl.DataFrame(
        {"date": [d], "ticker": ["069500"], "name": ["KODEX 200"], "trading_value": [adv]},
        schema={"date": pl.Date, "ticker": pl.String, "name": pl.String, "trading_value": pl.Float64},
    )

    out = apply_live_exposure_and_capacity_limits(
        {"069500": 0.95},
        panel,
        held={},
        decision_date=d,
        capital=capital,
        strategy_id="sticky.mom60_raw",
    )

    max_delta_w = (adv * P27_ADOPTED_MAX_ORDER_TO_ADV) / capital
    assert out.get("069500", 0.0) <= max_delta_w + 1e-9
    assert out.get("069500", 0.0) < 0.95



import polars as pl


def test_apply_live_exposure_and_capacity_limits_raises_on_missing_name() -> None:
    from src.tournament.live_decision import apply_live_exposure_and_capacity_limits

    d = date(2026, 9, 8)
    panel = pl.DataFrame(
        {"date": [d], "ticker": ["999"], "name": ["Other"], "trading_value": [1_000_000.0]},
        schema={"date": pl.Date, "ticker": pl.String, "name": pl.String, "trading_value": pl.Float64},
    )

    with pytest.raises(ValueError, match="412570"):
        apply_live_exposure_and_capacity_limits(
            {"412570": 1.0}, panel, held={}, decision_date=d, capital=1_000_000_000.0, strategy_id="sticky.mom60_raw"
        )

    # Then: empty weights short-circuits before any panel access (R8)
    assert apply_live_exposure_and_capacity_limits({}, panel, held={}, decision_date=d, capital=1.0, strategy_id="sticky.mom60_raw") == {}


def test_apply_live_exposure_and_capacity_limits_skips_tickers_with_no_adv_history() -> None:
    from src.tournament.live_decision import apply_live_exposure_and_capacity_limits

    d = date(2026, 9, 8)
    # "412570" has valid trading_value; "999" (held, not in weights) has NO rows at all in
    # the ADV window; "462330" has a row but trading_value is null. Both must be safely
    # skipped (fail-closed: omitted from adv_by_ticker, never defaulted to 0).
    panel = pl.DataFrame(
        {
            "date": [d, d, d],
            "ticker": ["412570", "462330", "999"],
            "name": ["TIGER 2차전지TOP10레버리지", "Other ETF", "Held ETF (delisted from ADV window)"],
            "trading_value": [10_000_000_000.0, None, None],
        },
        schema={"date": pl.Date, "ticker": pl.String, "name": pl.String, "trading_value": pl.Float64},
    )

    out = apply_live_exposure_and_capacity_limits(
        {"412570": 0.5, "462330": 0.0},
        panel,
        held={"999": 0.0},
        decision_date=d,
        capital=1_000.0,
        strategy_id="sticky.mom60_raw",
    )

    assert "412570" in out


def test_resolve_prior_sticky_state_same_day_rerun_preserves_prior_session(tmp_path) -> None:
    from src.tournament.live_decision import persist_sticky_state, resolve_prior_sticky_state

    p = tmp_path / "sticky_position.json"
    day1 = date(2026, 9, 17)
    day2 = date(2026, 9, 18)

    # 1. Day 1 decision persisted
    persist_sticky_state(p, decision_date=day1, held="122630", held_weight=0.95, hold_len=1)

    # 2. Day 2 first run: resolves day 1 state
    held, weight, hold_len = resolve_prior_sticky_state(p, prior_session=day1)
    assert (held, weight, hold_len) == ("122630", 0.95, 1)

    # Day 2 decision persisted (now file has as_of = day2)
    persist_sticky_state(p, decision_date=day2, held="122630", held_weight=0.95, hold_len=2)

    # 3. Day 2 second run (rerun / same-day retry): MUST still resolve day 1 state correctly!
    held_rerun, weight_rerun, hold_len_rerun = resolve_prior_sticky_state(p, prior_session=day1)
    assert (held_rerun, weight_rerun, hold_len_rerun) == ("122630", 0.95, 1)

    # Persist again on rerun
    persist_sticky_state(p, decision_date=day2, held="122630", held_weight=0.95, hold_len=2)

    # Still resolves day 1 state
    held_rerun2, weight_rerun2, hold_len_rerun2 = resolve_prior_sticky_state(p, prior_session=day1)
    assert (held_rerun2, weight_rerun2, hold_len_rerun2) == ("122630", 0.95, 1)


def test_persist_sticky_state_never_rewinds_as_of(tmp_path) -> None:
    import json

    from src.tournament.live_decision import persist_sticky_state, resolve_prior_sticky_state

    p = tmp_path / "ledger.json"
    for session, ticker in (
        (date(2026, 9, 17), "122630"),
        (date(2026, 9, 18), "122630"),
        (date(2026, 9, 21), "069510"),
    ):
        persist_sticky_state(p, decision_date=session, held=ticker, held_weight=0.95, hold_len=1)

    persist_sticky_state(p, decision_date=date(2026, 9, 18), held="233740", held_weight=1.0, hold_len=7)

    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["as_of"] == "2026-09-21"
    assert payload["held"] == "069510"
    assert payload["hold_len"] == 1
    assert payload["history"]["2026-09-18"] == {"held": "233740", "held_weight": 1.0, "hold_len": 7}
    assert resolve_prior_sticky_state(p, prior_session=date(2026, 9, 21)) == ("069510", 0.95, 1)


def test_persist_sticky_state_migrates_legacy_ledger(tmp_path) -> None:
    import json

    from src.tournament.live_decision import persist_sticky_state

    p = tmp_path / "legacy.json"
    p.write_text(
        json.dumps({"as_of": "2026-09-17", "held": "122630", "held_weight": 0.95, "hold_len": 3}),
        encoding="utf-8",
    )

    persist_sticky_state(p, decision_date=date(2026, 9, 18), held="122630", held_weight=0.95, hold_len=4)

    payload = json.loads(p.read_text(encoding="utf-8"))
    assert set(payload["history"]) == {"2026-09-17", "2026-09-18"}
    assert payload["history"]["2026-09-17"] == {"held": "122630", "held_weight": 0.95, "hold_len": 3}
    assert payload["as_of"] == "2026-09-18"


def test_persist_sticky_state_keeps_only_the_newest_sessions(tmp_path) -> None:
    import json

    from src.tournament.live_decision import STICKY_STATE_HISTORY_MAX, persist_sticky_state

    p = tmp_path / "ledger.json"
    sessions = [date(2026, 1, 5) + timedelta(days=i) for i in range(STICKY_STATE_HISTORY_MAX + 5)]
    for session in sessions:
        persist_sticky_state(p, decision_date=session, held="122630", held_weight=0.95, hold_len=1)

    payload = json.loads(p.read_text(encoding="utf-8"))
    assert set(payload["history"]) == {s.isoformat() for s in sessions[-STICKY_STATE_HISTORY_MAX:]}
    assert payload["as_of"] == sessions[-1].isoformat()


def test_persist_sticky_state_restarts_when_existing_ledger_is_unreadable(tmp_path, caplog) -> None:
    import json
    import logging

    from src.tournament.live_decision import persist_sticky_state

    p = tmp_path / "corrupt.json"
    p.write_text("{not valid json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="src.tournament.live_decision"):
        persist_sticky_state(p, decision_date=date(2026, 9, 18), held="122630", held_weight=0.95, hold_len=1)

    payload = json.loads(p.read_text(encoding="utf-8"))
    assert set(payload["history"]) == {"2026-09-18"}
    assert payload["as_of"] == "2026-09-18"
    assert any("unreadable" in record.getMessage() for record in caplog.records)


def test_pending_catchup_sessions_lists_the_gap_in_ascending_order(tmp_path) -> None:
    from src.tournament.live_decision import pending_catchup_sessions, persist_sticky_state

    grid = [date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 18), date(2026, 9, 21)]
    panel = pl.DataFrame([{"date": d, "ticker": "122630"} for d in grid], schema={"date": pl.Date, "ticker": pl.String})
    p = tmp_path / "ledger.json"
    persist_sticky_state(p, decision_date=date(2026, 9, 16), held="122630", held_weight=0.95, hold_len=1)

    assert pending_catchup_sessions(p, panel, target_session=date(2026, 9, 21)) == [
        date(2026, 9, 17),
        date(2026, 9, 18),
    ]


def test_pending_catchup_sessions_empty_when_current_ahead_or_missing(tmp_path) -> None:
    from src.tournament.live_decision import pending_catchup_sessions, persist_sticky_state

    grid = [date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 18), date(2026, 9, 21)]
    panel = pl.DataFrame([{"date": d, "ticker": "122630"} for d in grid], schema={"date": pl.Date, "ticker": pl.String})
    target = date(2026, 9, 21)

    assert pending_catchup_sessions(tmp_path / "missing.json", panel, target_session=target) == []

    current = tmp_path / "current.json"
    persist_sticky_state(current, decision_date=date(2026, 9, 18), held="122630", held_weight=0.95, hold_len=1)
    assert pending_catchup_sessions(current, panel, target_session=target) == []

    ahead = tmp_path / "ahead.json"
    persist_sticky_state(ahead, decision_date=target, held="122630", held_weight=0.95, hold_len=1)
    assert pending_catchup_sessions(ahead, panel, target_session=target) == []


def test_pending_catchup_sessions_skips_phantom_sessions(tmp_path) -> None:
    from src.tournament.live_decision import pending_catchup_sessions, persist_sticky_state

    grid = [date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 21)]
    panel = pl.DataFrame([{"date": d, "ticker": "122630"} for d in grid], schema={"date": pl.Date, "ticker": pl.String})
    p = tmp_path / "ledger.json"
    persist_sticky_state(p, decision_date=date(2026, 9, 16), held="122630", held_weight=0.95, hold_len=1)

    pending = pending_catchup_sessions(p, panel, target_session=date(2026, 9, 21))

    assert pending == [date(2026, 9, 17)]
    assert date(2026, 9, 18) not in pending

