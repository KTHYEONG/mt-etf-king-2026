"""Invariant guards for the contest segment policy core domain."""

from __future__ import annotations

import ast
import json
import math
from datetime import date, time
from pathlib import Path
from typing import Any

import pytest

from src.contest.daily import QuoteBar
from src.contest.leaderboard import LeaderboardEntry, LeaderboardSnapshot
from src.contest.segment import (
    EquitySource,
    IntradaySide,
    LedgerState,
    SegmentAction,
    SegmentCard,
    SegmentSettings,
    Standing,
    TradingWindow,
    advance_ledger,
    bar_change_pct,
    build_segment_card,
    decide_intraday_side,
    executed_side_from_card,
    leaderboard_equity,
    ledger_from_dict,
    ledger_to_dict,
    render_segment_markdown,
    segment_to_dict,
    session_policy_multiple,
    compute_standing,
)
from src.core.calendar import get_calendar

CAL = get_calendar()


def _fixture_settings_dict() -> dict[str, Any]:
    return {
        "long_ticker": "0193T0",
        "long_alias": "HY2",
        "inverse_ticker": "0197X0",
        "inverse_alias": "HY2I",
        "vote": {
            "HY2": "0193T0",
            "SEMI2": "494310",
            "K2": "122630",
        },
        "vote_min_down": 2,
        "open_window": "09:05-09:15",
        "close_window": "15:10-15:19",
        "cost_bps_round_trip": 12.0,
        "prize_rank": 3,
        "endgame_sessions": 5,
        "ledger_state_name": "contest_ledger.json",
        "output_dir": "results/contest_segment",
    }


def _fixture_config(**overrides: Any) -> dict[str, Any]:
    seg = _fixture_settings_dict()
    if "segment" in overrides:
        seg = overrides.pop("segment")
    cfg: dict[str, Any] = {
        "end_date": "2026-11-13",
        "segment": seg,
    }
    cfg.update(overrides)
    return cfg


def _fixture_settings(**overrides: Any) -> SegmentSettings:
    seg = _fixture_settings_dict()
    seg.update(overrides)
    return SegmentSettings.load({"segment": seg, "end_date": "2026-11-13"})


def _bar(ticker: str, prev: float, open_: float, close: float) -> QuoteBar:
    return QuoteBar(ticker=ticker, prev_close=prev, open=open_, close=close)


# --- Settings Invariants ---


def test_settings_load_round_trip() -> None:
    """Settings load preserves vote config order and parses windows into time objects."""
    settings = SegmentSettings.load(_fixture_config())
    assert settings.vote == (("HY2", "0193T0"), ("SEMI2", "494310"), ("K2", "122630"))
    assert settings.open_window == TradingWindow(start=time(9, 5), end=time(9, 15))
    assert settings.close_window == TradingWindow(start=time(15, 10), end=time(15, 19))
    assert settings.cost_bps_round_trip == pytest.approx(12.0)
    assert settings.prize_rank == 3
    assert settings.endgame_sessions == 5
    assert settings.output_dir == Path("results/contest_segment")


@pytest.mark.parametrize(
    ("open_win", "close_win"),
    [
        ("08:55-09:10", "15:10-15:19"),
        ("09:05-09:15", "15:15-15:25"),
        ("09:00-09:10", "15:10-15:19"),
        ("09:05-09:15", "15:10-15:20"),
    ],
)
def test_settings_reject_auction_windows(open_win: str, close_win: str) -> None:
    """Settings reject windows outside continuous session boundaries (09:01-15:19)."""
    with pytest.raises(ValueError, match="outside continuous session limits"):
        _fixture_settings(open_window=open_win, close_window=close_win)


def test_settings_reject_overlapping_windows() -> None:
    """Settings reject open_window ending after close_window starts."""
    with pytest.raises(ValueError, match=">"):
        _fixture_settings(open_window="09:05-12:00", close_window="11:30-15:19")


@pytest.mark.parametrize("bad_vmd", [0, 4, True, "2", -1])
def test_settings_reject_bad_vote_threshold(bad_vmd: Any) -> None:
    """Settings reject vote_min_down outside [1, len(vote)] or non-int."""
    with pytest.raises(ValueError, match="vote_min_down"):
        _fixture_settings(vote_min_down=bad_vmd)


def test_settings_reject_duplicate_or_blank_tickers() -> None:
    """Settings reject duplicate tickers or equal long/inverse tickers or blanks."""
    with pytest.raises(ValueError, match="duplicate tickers"):
        _fixture_settings(vote={"A": "0193T0", "B": "0193T0"})
    with pytest.raises(ValueError, match="must differ"):
        _fixture_settings(long_ticker="0193T0", inverse_ticker="0193T0")
    with pytest.raises(ValueError, match="must differ"):
        _fixture_settings(long_alias="HY2", inverse_alias="HY2")
    with pytest.raises(ValueError, match="must not be blank"):
        _fixture_settings(long_ticker="   ")


def test_settings_missing_block() -> None:
    """Settings load raises KeyError when segment block is absent or non-mapping."""
    with pytest.raises(KeyError, match="segment"):
        SegmentSettings.load({"end_date": "2026-11-13"})
    with pytest.raises(KeyError, match="segment"):
        SegmentSettings.load({"end_date": "2026-11-13", "segment": "invalid"})
    seg = _fixture_settings_dict()
    del seg["long_ticker"]
    with pytest.raises(KeyError, match="missing required key"):
        SegmentSettings.load({"segment": seg})


def test_settings_validation_edge_cases() -> None:
    """Settings reject invalid windows, empty/non-mapping vote, bad costs and non-positive parameters."""
    with pytest.raises(ValueError, match="window must be"):
        _fixture_settings(open_window=123)
    with pytest.raises(ValueError, match="malformed window string"):
        _fixture_settings(open_window="09:05")
    with pytest.raises(ValueError, match="malformed window format"):
        _fixture_settings(open_window="9:05-15:10")
    with pytest.raises(ValueError, match="malformed window time"):
        _fixture_settings(open_window="25:00-15:10")
    with pytest.raises(ValueError, match="must precede end"):
        _fixture_settings(open_window="10:00-09:30")

    with pytest.raises(ValueError, match="vote must be a non-empty mapping"):
        _fixture_settings(vote="bad")
    with pytest.raises(ValueError, match="vote must be a non-empty mapping"):
        _fixture_settings(vote={})
    with pytest.raises(ValueError, match="vote aliases and tickers must not be blank"):
        _fixture_settings(vote={"": "0193T0"})
    with pytest.raises(ValueError, match="vote aliases and tickers must not be blank"):
        _fixture_settings(vote={"HY2": ""})

    with pytest.raises(ValueError, match="cost_bps_round_trip must be numeric"):
        _fixture_settings(cost_bps_round_trip=True)
    with pytest.raises(ValueError, match="cost_bps_round_trip must be numeric"):
        _fixture_settings(cost_bps_round_trip="abc")
    with pytest.raises(ValueError, match="cost_bps_round_trip must be finite and in"):
        _fixture_settings(cost_bps_round_trip=-1.0)
    with pytest.raises(ValueError, match="cost_bps_round_trip must be finite and in"):
        _fixture_settings(cost_bps_round_trip=100.0)

    with pytest.raises(ValueError, match="prize_rank must be a positive int"):
        _fixture_settings(prize_rank=0)
    with pytest.raises(ValueError, match="endgame_sessions must be a positive int"):
        _fixture_settings(endgame_sessions=-1)

    with pytest.raises(ValueError, match="ledger_state_name and output_dir must not be blank"):
        _fixture_settings(ledger_state_name="")
    with pytest.raises(ValueError, match="ledger_state_name and output_dir must not be blank"):
        _fixture_settings(output_dir="")


# --- Vote Invariants ---


def test_majority_down_votes_long() -> None:
    """Majority down votes long."""
    settings = _fixture_settings()
    changes = {"0193T0": -5.3, "494310": -8.1, "122630": 0.2}
    side, down_votes = decide_intraday_side(changes, settings)
    assert side is IntradaySide.LONG
    assert down_votes == 2


def test_majority_up_votes_short() -> None:
    """Majority up votes short."""
    settings = _fixture_settings()
    changes = {"0193T0": 1.0, "494310": -0.5, "122630": 0.3}
    side, down_votes = decide_intraday_side(changes, settings)
    assert side is IntradaySide.SHORT
    assert down_votes == 1


def test_flat_counts_as_not_down() -> None:
    """Flat change (0.0) counts as not-down."""
    settings = _fixture_settings()
    changes = {"0193T0": 0.0, "494310": 0.0, "122630": -1.0}
    side, down_votes = decide_intraday_side(changes, settings)
    assert side is IntradaySide.SHORT
    assert down_votes == 1


def test_partial_vote_is_fail_closed() -> None:
    """Missing ticker fail-closes decision to None while tracking valid down votes."""
    settings = _fixture_settings()
    changes = {"0193T0": -5.0, "494310": -3.0}  # "122630" missing
    side, down_votes = decide_intraday_side(changes, settings)
    assert side is None
    assert down_votes == 2


def test_non_finite_vote_is_fail_closed() -> None:
    """NaN change fail-closes decision to None."""
    settings = _fixture_settings()
    changes = {"0193T0": -5.0, "494310": -3.0, "122630": float("nan")}
    side, down_votes = decide_intraday_side(changes, settings)
    assert side is None
    assert down_votes == 2


# --- Card Invariants ---


def test_card_for_session_holds_long() -> None:
    """Card for 2026-10-07 holds long when vote majority is down."""
    cfg = _fixture_config()
    session = date(2026, 10, 7)
    bars = {
        "0193T0": _bar("0193T0", 10700.0, 10500.0, 10130.0),
        "494310": _bar("494310", 107115.0, 105000.0, 98400.0),
        "122630": _bar("122630", 111805.0, 110000.0, 107230.0),
    }
    standing = Standing(
        snapshot_date=session,
        our_equity=1.0,
        our_source=EquitySource.LEDGER,
        our_as_of=session,
        prize_line_equity=1.2,
        leader_equity=1.5,
        required_multiple=1.2,
    )
    card = build_segment_card(session, CAL, bars, cfg, standing)
    assert card.action is SegmentAction.HOLD_LONG
    assert card.execution_session == date(2026, 10, 8)
    assert card.intraday_side is IntradaySide.LONG
    assert card.down_votes == 3
    assert not card.final_session


def test_card_switches_short_on_up_majority() -> None:
    """Card switches short when 2 of 3 vote bars are up."""
    cfg = _fixture_config()
    session = date(2026, 10, 7)
    bars = {
        "0193T0": _bar("0193T0", 10000.0, 10100.0, 10200.0),
        "494310": _bar("494310", 100000.0, 101000.0, 102000.0),
        "122630": _bar("122630", 100000.0, 99000.0, 98000.0),
    }
    standing = Standing(None, None, None, None, None, None, None)
    card = build_segment_card(session, CAL, bars, cfg, standing)
    assert card.action is SegmentAction.SWITCH_SHORT
    assert card.intraday_side is IntradaySide.SHORT
    assert card.down_votes == 1


def test_missing_vote_bar_yields_no_data() -> None:
    """Missing vote bar yields NO_DATA with QUOTES_MISSING and side None."""
    cfg = _fixture_config()
    session = date(2026, 10, 7)
    bars = {
        "0193T0": _bar("0193T0", 10700.0, 10500.0, 10130.0),
        "494310": _bar("494310", 107115.0, 105000.0, 98400.0),
        # 122630 missing
    }
    standing = Standing(None, None, None, None, None, None, None)
    card = build_segment_card(session, CAL, bars, cfg, standing)
    assert card.action is SegmentAction.NO_DATA
    assert card.intraday_side is None
    assert "QUOTES_MISSING" in card.warnings


def test_non_positive_bar_yields_no_data() -> None:
    """Non-positive close price yields NO_DATA."""
    cfg = _fixture_config()
    session = date(2026, 10, 7)
    bars = {
        "0193T0": _bar("0193T0", 10700.0, 10500.0, 10130.0),
        "494310": _bar("494310", 107115.0, 105000.0, 98400.0),
        "122630": _bar("122630", 111805.0, 110000.0, 0.0),
    }
    standing = Standing(None, None, None, None, None, None, None)
    card = build_segment_card(session, CAL, bars, cfg, standing)
    assert card.action is SegmentAction.NO_DATA
    assert card.intraday_side is None
    assert "QUOTES_MISSING" in card.warnings


def test_contest_end_yields_contest_over() -> None:
    """Session on or after end_date yields CONTEST_OVER."""
    cfg = _fixture_config(end_date="2026-11-13")
    session = date(2026, 11, 13)
    standing = Standing(None, None, None, None, None, None, None)
    card = build_segment_card(session, CAL, {}, cfg, standing)
    assert card.action is SegmentAction.CONTEST_OVER
    assert card.execution_session is None
    assert card.down_votes == 0
    assert card.intraday_side is None


def test_final_session_flagged() -> None:
    """Execution on contest end_date flags final_session and FINAL_SESSION warning."""
    cfg = _fixture_config(end_date="2026-11-13")
    prev_to_end = CAL.previous_session(date(2026, 11, 13))
    bars = {
        "0193T0": _bar("0193T0", 100.0, 100.0, 99.0),
        "494310": _bar("494310", 100.0, 100.0, 99.0),
        "122630": _bar("122630", 100.0, 100.0, 99.0),
    }
    standing = Standing(None, None, None, None, None, None, None)
    card = build_segment_card(prev_to_end, CAL, bars, cfg, standing)
    assert card.execution_session == date(2026, 11, 13)
    assert card.final_session is True
    assert card.sessions_remaining_after_execution == 0
    assert "FINAL_SESSION" in card.warnings


def test_endgame_window_flagged() -> None:
    """Card flags endgame when remaining sessions < endgame_sessions."""
    cfg = _fixture_config(end_date="2026-11-13")
    sessions = CAL.sessions(date(2026, 11, 1), date(2026, 11, 13))
    # Pick a session such that execution has fewer than 5 sessions remaining
    exec_session = sessions[-5]  # remaining after exec: 4 sessions
    signal_session = CAL.previous_session(exec_session)
    bars = {
        "0193T0": _bar("0193T0", 100.0, 100.0, 99.0),
        "494310": _bar("494310", 100.0, 100.0, 99.0),
        "122630": _bar("122630", 100.0, 100.0, 99.0),
    }
    standing = Standing(None, None, None, None, None, None, None)
    card = build_segment_card(signal_session, CAL, bars, cfg, standing)
    assert card.sessions_remaining_after_execution == 4
    assert card.endgame is True
    assert "ENDGAME_REVIEW" in card.warnings


def test_calendar_failure_is_fail_closed() -> None:
    """Calendar failure fail-closes card with NO_DATA and CALENDAR_UNAVAILABLE."""

    class _BadCal:
        def next_session(self, d: date) -> date:
            raise ValueError("calendar boom")

    cfg = _fixture_config()
    standing = Standing(None, None, None, None, None, None, None)
    card = build_segment_card(date(2026, 10, 7), _BadCal(), {}, cfg, standing)
    assert card.action is SegmentAction.NO_DATA
    assert card.execution_session is None
    assert "CALENDAR_UNAVAILABLE" in card.warnings


def test_next_session_after_end_date_yields_contest_over() -> None:
    """When next session lies after end_date, card is CONTEST_OVER."""

    class _SkippingCal:
        def next_session(self, d: date) -> date:
            return date(2026, 11, 14)

    cfg = _fixture_config(end_date="2026-11-13")
    standing = Standing(None, None, None, None, None, None, None)
    card = build_segment_card(date(2026, 11, 12), _SkippingCal(), {}, cfg, standing)
    assert card.action is SegmentAction.CONTEST_OVER
    assert card.execution_session is None


def test_calendar_sessions_failure_handled() -> None:
    """A calendar.sessions failure fail-closes the card instead of silently reporting zero sessions remaining."""

    class _FailingSessionsCal:
        def next_session(self, d: date) -> date:
            return date(2026, 10, 8)

        def sessions(self, start: date, end: date) -> list[date]:
            raise ValueError("sessions failure")

    cfg = _fixture_config()
    bars = {
        "0193T0": _bar("0193T0", 100.0, 100.0, 99.0),
        "494310": _bar("494310", 100.0, 100.0, 99.0),
        "122630": _bar("122630", 100.0, 100.0, 99.0),
    }
    standing = Standing(None, None, None, None, None, None, None)
    card = build_segment_card(date(2026, 10, 7), _FailingSessionsCal(), bars, cfg, standing)
    assert card.action is SegmentAction.NO_DATA
    assert card.intraday_side is None
    assert card.execution_session is None
    assert "CALENDAR_UNAVAILABLE" in card.warnings


def test_extra_warnings_appended_without_duplicates() -> None:
    """Extra warnings are appended preserving order and deduplicating card warnings."""
    cfg = _fixture_config(end_date="2026-11-13")
    sessions = CAL.sessions(date(2026, 11, 1), date(2026, 11, 13))
    exec_session = sessions[-5]
    signal_session = CAL.previous_session(exec_session)
    bars = {
        "0193T0": _bar("0193T0", 100.0, 100.0, 99.0),
        "494310": _bar("494310", 100.0, 100.0, 99.0),
        "122630": _bar("122630", 100.0, 100.0, 99.0),
    }
    standing = Standing(None, None, None, None, None, None, None)
    card = build_segment_card(
        signal_session, CAL, bars, cfg, standing, extra_warnings=("LEDGER_STALE", "ENDGAME_REVIEW")
    )
    assert card.warnings == ("ENDGAME_REVIEW", "LEDGER_STALE")


# --- Executed Side and Multiple Invariants ---


def test_executed_side_mapping() -> None:
    """executed_side_from_card correctly maps actions and handles missing values."""
    assert executed_side_from_card({"action": SegmentAction.HOLD_LONG}) is IntradaySide.LONG
    assert executed_side_from_card({"action": "HOLD_LONG"}) is IntradaySide.LONG
    assert executed_side_from_card({"action": SegmentAction.NO_DATA}) is None
    assert executed_side_from_card({"action": "NO_DATA"}) is None
    assert executed_side_from_card({"action": SegmentAction.SWITCH_SHORT}) is IntradaySide.SHORT
    assert executed_side_from_card({"action": "SWITCH_SHORT"}) is IntradaySide.SHORT
    assert executed_side_from_card({"action": SegmentAction.CONTEST_OVER}) is None
    assert executed_side_from_card({"action": "CONTEST_OVER"}) is None
    assert executed_side_from_card({"action": "UNKNOWN"}) is None
    assert executed_side_from_card(None) is None


def test_long_session_multiple() -> None:
    """Long session multiple equals overnight * intraday of long vehicle without costs."""
    settings = _fixture_settings()
    bars = {"0193T0": _bar("0193T0", 100.0, 102.0, 99.0)}
    mult = session_policy_multiple(bars, IntradaySide.LONG, False, settings)
    assert mult is not None
    assert mult == pytest.approx(0.99, abs=1e-12)


def test_short_session_multiple_pays_two_round_trips() -> None:
    """Non-final short session pays two round trips (cost * 2)."""
    settings = _fixture_settings(cost_bps_round_trip=12.0)
    bars = {
        "0193T0": _bar("0193T0", 100.0, 102.0, 100.0),
        "0197X0": _bar("0197X0", 50.0, 50.0, 52.0),
    }
    mult = session_policy_multiple(bars, IntradaySide.SHORT, False, settings)
    assert mult is not None
    expected = 1.02 * 1.04 * (1.0 - 0.0024)
    assert mult == pytest.approx(expected, abs=1e-12)


def test_final_short_session_pays_one_round_trip() -> None:
    """Final short session pays one round trip (cost * 1)."""
    settings = _fixture_settings(cost_bps_round_trip=12.0)
    bars = {
        "0193T0": _bar("0193T0", 100.0, 102.0, 100.0),
        "0197X0": _bar("0197X0", 50.0, 50.0, 52.0),
    }
    mult = session_policy_multiple(bars, IntradaySide.SHORT, True, settings)
    assert mult is not None
    expected = 1.02 * 1.04 * (1.0 - 0.0012)
    assert mult == pytest.approx(expected, abs=1e-12)


def test_short_multiple_needs_inverse_bar() -> None:
    """Short multiple returns None when inverse bar is absent."""
    settings = _fixture_settings()
    bars = {"0193T0": _bar("0193T0", 100.0, 102.0, 100.0)}
    assert session_policy_multiple(bars, IntradaySide.SHORT, False, settings) is None


def test_session_policy_multiple_invalid_long_bar() -> None:
    """session_policy_multiple returns None if long bar is absent or non-positive."""
    settings = _fixture_settings()
    assert session_policy_multiple({}, IntradaySide.LONG, False, settings) is None
    bars = {"0193T0": _bar("0193T0", 0.0, 100.0, 100.0)}
    assert session_policy_multiple(bars, IntradaySide.LONG, False, settings) is None


# --- Ledger Invariants ---


def test_user_anchor_wins() -> None:
    """Valid user equity anchor overrides leaderboard and compounding."""
    settings = _fixture_settings()
    prev = LedgerState(as_of=date(2026, 10, 6), equity=1.0, source=EquitySource.LEDGER)
    session = date(2026, 10, 7)
    state, warns = advance_ledger(
        prev=prev,
        session=session,
        prev_session=date(2026, 10, 6),
        executed_side=IntradaySide.LONG,
        bars={},
        session_is_final=False,
        leaderboard_equity=1.10,
        user_equity=0.90,
        settings=settings,
    )
    assert state == LedgerState(as_of=session, equity=0.90, source=EquitySource.USER)
    assert warns == ()


def test_leaderboard_anchor_beats_compounding() -> None:
    """Leaderboard anchor is used when user anchor is absent."""
    settings = _fixture_settings()
    prev = LedgerState(as_of=date(2026, 10, 6), equity=1.0, source=EquitySource.LEDGER)
    session = date(2026, 10, 7)
    state, warns = advance_ledger(
        prev=prev,
        session=session,
        prev_session=date(2026, 10, 6),
        executed_side=IntradaySide.LONG,
        bars={},
        session_is_final=False,
        leaderboard_equity=1.10,
        user_equity=None,
        settings=settings,
    )
    assert state == LedgerState(as_of=session, equity=1.10, source=EquitySource.LEADERBOARD)
    assert warns == ()


def test_compounding_from_previous_session() -> None:
    """Compounding advances ledger when no anchors exist and bars are valid."""
    settings = _fixture_settings()
    prev = LedgerState(as_of=date(2026, 10, 6), equity=1.05, source=EquitySource.LEDGER)
    session = date(2026, 10, 7)
    bars = {"0193T0": _bar("0193T0", 100.0, 102.0, 99.0)}
    state, warns = advance_ledger(
        prev=prev,
        session=session,
        prev_session=date(2026, 10, 6),
        executed_side=IntradaySide.LONG,
        bars=bars,
        session_is_final=False,
        leaderboard_equity=None,
        user_equity=None,
        settings=settings,
    )
    assert state is not None
    assert state.source is EquitySource.LEDGER
    assert state.as_of == session
    assert state.equity == pytest.approx(1.05 * 0.99, abs=1e-12)
    assert warns == ()


def test_rerun_does_not_compound_twice() -> None:
    """Rerun on the same session returns prev unchanged with no warning."""
    settings = _fixture_settings()
    session = date(2026, 10, 7)
    prev = LedgerState(as_of=session, equity=1.05, source=EquitySource.LEDGER)
    state, warns = advance_ledger(
        prev=prev,
        session=session,
        prev_session=date(2026, 10, 6),
        executed_side=IntradaySide.LONG,
        bars={},
        session_is_final=False,
        leaderboard_equity=None,
        user_equity=None,
        settings=settings,
    )
    assert state == prev
    assert warns == ()


def test_ledger_never_moves_backwards() -> None:
    """Ledger state with as_of > session returns unchanged with LEDGER_AHEAD."""
    settings = _fixture_settings()
    prev = LedgerState(as_of=date(2026, 10, 8), equity=1.05, source=EquitySource.LEDGER)
    session = date(2026, 10, 7)
    state, warns = advance_ledger(
        prev=prev,
        session=session,
        prev_session=date(2026, 10, 6),
        executed_side=IntradaySide.LONG,
        bars={},
        session_is_final=False,
        leaderboard_equity=1.2,
        user_equity=1.2,
        settings=settings,
    )
    assert state == prev
    assert warns == ("LEDGER_AHEAD",)


def test_gap_leaves_ledger_stale() -> None:
    """Gap in sessions leaves ledger stale without updating as_of."""
    settings = _fixture_settings()
    prev = LedgerState(as_of=date(2026, 10, 5), equity=1.05, source=EquitySource.LEDGER)
    session = date(2026, 10, 7)
    state, warns = advance_ledger(
        prev=prev,
        session=session,
        prev_session=date(2026, 10, 6),
        executed_side=IntradaySide.LONG,
        bars={},
        session_is_final=False,
        leaderboard_equity=None,
        user_equity=None,
        settings=settings,
    )
    assert state == prev
    assert warns == ("LEDGER_STALE",)


def test_unknown_side_leaves_ledger_stale() -> None:
    """executed_side None leaves ledger stale."""
    settings = _fixture_settings()
    prev = LedgerState(as_of=date(2026, 10, 6), equity=1.05, source=EquitySource.LEDGER)
    session = date(2026, 10, 7)
    state, warns = advance_ledger(
        prev=prev,
        session=session,
        prev_session=date(2026, 10, 6),
        executed_side=None,
        bars={},
        session_is_final=False,
        leaderboard_equity=None,
        user_equity=None,
        settings=settings,
    )
    assert state == prev
    assert warns == ("LEDGER_STALE",)


def test_advance_ledger_missing_bars_leaves_stale() -> None:
    """advance_ledger leaves ledger stale when compounding fails due to missing bars."""
    settings = _fixture_settings()
    prev = LedgerState(as_of=date(2026, 10, 6), equity=1.05, source=EquitySource.LEDGER)
    session = date(2026, 10, 7)
    state, warns = advance_ledger(
        prev=prev,
        session=session,
        prev_session=date(2026, 10, 6),
        executed_side=IntradaySide.LONG,
        bars={},
        session_is_final=False,
        leaderboard_equity=None,
        user_equity=None,
        settings=settings,
    )
    assert state == prev
    assert warns == ("LEDGER_STALE",)


def test_invalid_user_anchor_ignored() -> None:
    """Non-finite or negative user anchor is ignored with USER_EQUITY_INVALID warning."""
    settings = _fixture_settings()
    session = date(2026, 10, 7)
    state, warns = advance_ledger(
        prev=None,
        session=session,
        prev_session=date(2026, 10, 6),
        executed_side=None,
        bars={},
        session_is_final=False,
        leaderboard_equity=1.12,
        user_equity=-0.1,
        settings=settings,
    )
    assert state == LedgerState(as_of=session, equity=1.12, source=EquitySource.LEADERBOARD)
    assert warns == ("USER_EQUITY_INVALID",)


def test_no_anchor_and_no_history() -> None:
    """No anchor and no previous state yields (None, ('EQUITY_UNKNOWN',))."""
    settings = _fixture_settings()
    session = date(2026, 10, 7)
    state, warns = advance_ledger(
        prev=None,
        session=session,
        prev_session=date(2026, 10, 6),
        executed_side=None,
        bars={},
        session_is_final=False,
        leaderboard_equity=None,
        user_equity=None,
        settings=settings,
    )
    assert state is None
    assert warns == ("EQUITY_UNKNOWN",)


def test_ledger_dict_round_trip_and_rejection() -> None:
    """Ledger state round-trips via to_dict/from_dict; rejects malformed inputs."""
    state = LedgerState(as_of=date(2026, 10, 7), equity=1.054, source=EquitySource.LEDGER)
    raw = ledger_to_dict(state)
    restored = ledger_from_dict(raw)
    assert restored == state

    assert ledger_from_dict("not-a-dict") is None
    assert ledger_from_dict({"as_of": "invalid", "equity": 1.0, "source": "LEDGER"}) is None
    assert ledger_from_dict({"as_of": "2026-10-07", "equity": 0.0, "source": "LEDGER"}) is None
    assert ledger_from_dict({"as_of": "2026-10-07", "equity": -1.0, "source": "LEDGER"}) is None
    assert ledger_from_dict({"as_of": "2026-10-07", "equity": float("nan"), "source": "LEDGER"}) is None
    assert ledger_from_dict({"as_of": "2026-10-07", "equity": 1.0, "source": "INVALID_SRC"}) is None


# --- Standing Invariants ---


def test_prize_line_excludes_us() -> None:
    """Prize line calculation excludes our entry."""
    session = date(2026, 10, 7)
    entries = (
        LeaderboardEntry(rank=1, user_name="alice", total_return_pct=50.0, daily_return_pct=1.0),
        LeaderboardEntry(rank=2, user_name="our_nick", total_return_pct=40.0, daily_return_pct=1.0),
        LeaderboardEntry(rank=3, user_name="bob", total_return_pct=30.0, daily_return_pct=1.0),
        LeaderboardEntry(rank=4, user_name="charlie", total_return_pct=20.0, daily_return_pct=1.0),
        LeaderboardEntry(rank=5, user_name="david", total_return_pct=10.0, daily_return_pct=1.0),
    )
    snap = LeaderboardSnapshot(base_date=session, requested_at="", entries=entries, purchases={})
    ledger = LedgerState(as_of=session, equity=1.40, source=EquitySource.LEADERBOARD)

    standing, warns = compute_standing(snap, session, "our_nick", ledger, prize_rank=3)
    # Competitors excluding us: alice (50%), bob (30%), charlie (20%), david (10%)
    # Leader is alice: 1.50
    # 3rd best is charlie: 1.20
    assert standing.leader_equity == pytest.approx(1.50)
    assert standing.prize_line_equity == pytest.approx(1.20)
    assert standing.required_multiple == pytest.approx(1.20 / 1.40)
    assert warns == ()


def test_required_multiple() -> None:
    """required_multiple equals prize_line_equity / our_equity."""
    session = date(2026, 10, 7)
    entries = (
        LeaderboardEntry(rank=1, user_name="c1", total_return_pct=100.0, daily_return_pct=0.0),
        LeaderboardEntry(rank=2, user_name="c2", total_return_pct=80.0, daily_return_pct=0.0),
        LeaderboardEntry(rank=3, user_name="c3", total_return_pct=70.0, daily_return_pct=0.0),
    )
    snap = LeaderboardSnapshot(base_date=session, requested_at="", entries=entries, purchases={})
    ledger = LedgerState(as_of=session, equity=0.885, source=EquitySource.USER)
    standing, warns = compute_standing(snap, session, "our_nick", ledger, prize_rank=3)
    assert standing.prize_line_equity == pytest.approx(1.70)
    assert standing.required_multiple == pytest.approx(1.70 / 0.885, abs=1e-12)
    assert warns == ()


def test_stale_snapshot_not_mixed() -> None:
    """Stale snapshot produces LEADERBOARD_STALE warning and None competitor equities."""
    session = date(2026, 10, 7)
    snap = LeaderboardSnapshot(
        base_date=date(2026, 10, 6),
        requested_at="",
        entries=(LeaderboardEntry(rank=1, user_name="c1", total_return_pct=20.0, daily_return_pct=0.0),),
        purchases={},
    )
    ledger = LedgerState(as_of=session, equity=1.0, source=EquitySource.LEDGER)
    standing, warns = compute_standing(snap, session, "our_nick", ledger, prize_rank=3)
    assert standing.prize_line_equity is None
    assert standing.leader_equity is None
    assert standing.required_multiple is None
    assert "LEADERBOARD_STALE" in warns


def test_stale_ledger_reported() -> None:
    """Stale ledger produces LEDGER_STALE warning and None required_multiple."""
    session = date(2026, 10, 7)
    entries = (
        LeaderboardEntry(rank=1, user_name="c1", total_return_pct=50.0, daily_return_pct=0.0),
        LeaderboardEntry(rank=2, user_name="c2", total_return_pct=30.0, daily_return_pct=0.0),
        LeaderboardEntry(rank=3, user_name="c3", total_return_pct=20.0, daily_return_pct=0.0),
    )
    snap = LeaderboardSnapshot(base_date=session, requested_at="", entries=entries, purchases={})
    ledger = LedgerState(as_of=date(2026, 10, 6), equity=1.0, source=EquitySource.LEDGER)
    standing, warns = compute_standing(snap, session, "our_nick", ledger, prize_rank=3)
    assert standing.required_multiple is None
    assert "LEDGER_STALE" in warns


def test_standing_with_no_ledger() -> None:
    """compute_standing adds EQUITY_UNKNOWN when ledger is None."""
    session = date(2026, 10, 7)
    standing, warns = compute_standing(None, session, "our_nick", None, prize_rank=3)
    assert standing.our_equity is None
    assert "EQUITY_UNKNOWN" in warns


def test_too_few_competitors() -> None:
    """prize_line_equity is None when fewer competitors than prize_rank exist."""
    session = date(2026, 10, 7)
    entries = (
        LeaderboardEntry(rank=1, user_name="c1", total_return_pct=50.0, daily_return_pct=0.0),
        LeaderboardEntry(rank=2, user_name="c2", total_return_pct=30.0, daily_return_pct=0.0),
    )
    snap = LeaderboardSnapshot(base_date=session, requested_at="", entries=entries, purchases={})
    ledger = LedgerState(as_of=session, equity=1.0, source=EquitySource.LEDGER)
    standing, warns = compute_standing(snap, session, "our_nick", ledger, prize_rank=3)
    assert standing.leader_equity == pytest.approx(1.50)
    assert standing.prize_line_equity is None
    assert standing.required_multiple is None


def test_leaderboard_equity_only_for_same_day_listing() -> None:
    """leaderboard_equity extracts equity only for same-day listing of nickname."""
    session = date(2026, 10, 7)
    snap_today = LeaderboardSnapshot(
        base_date=session,
        requested_at="",
        entries=(LeaderboardEntry(rank=1, user_name="other", total_return_pct=10.0, daily_return_pct=0.0),),
        purchases={},
    )
    snap_past = LeaderboardSnapshot(
        base_date=date(2026, 10, 6),
        requested_at="",
        entries=(LeaderboardEntry(rank=1, user_name="our_nick", total_return_pct=10.0, daily_return_pct=0.0),),
        purchases={},
    )
    assert leaderboard_equity(snap_today, session, "our_nick") is None
    assert leaderboard_equity(snap_past, session, "our_nick") is None


def test_leaderboard_equity_non_positive_return() -> None:
    """leaderboard_equity returns None if total_return_pct is non-finite or equity <= 0."""
    session = date(2026, 10, 7)
    snap = LeaderboardSnapshot(
        base_date=session,
        requested_at="",
        entries=(
            LeaderboardEntry(rank=1, user_name="our_nick", total_return_pct=-110.0, daily_return_pct=0.0),
            LeaderboardEntry(rank=2, user_name="nan_nick", total_return_pct=float("nan"), daily_return_pct=0.0),
        ),
        purchases={},
    )
    assert leaderboard_equity(snap, session, "our_nick") is None
    assert leaderboard_equity(snap, session, "nan_nick") is None


# --- Rendering Invariants ---


def test_short_checklist_timing_and_order() -> None:
    """Non-final short card renders open/close windows with correct sell/buy order and auction warning."""
    settings = _fixture_settings()
    standing = Standing(None, None, None, None, None, None, None)
    card = SegmentCard(
        session=date(2026, 10, 7),
        execution_session=date(2026, 10, 8),
        final_session=False,
        action=SegmentAction.SWITCH_SHORT,
        intraday_side=IntradaySide.SHORT,
        vote_changes=(("HY2", 1.0), ("SEMI2", 1.0), ("K2", -1.0)),
        down_votes=1,
        vote_min_down=2,
        sessions_remaining_after_execution=10,
        endgame=False,
        standing=standing,
        warnings=(),
    )
    md = render_segment_markdown(card, settings)
    assert "09:05–09:15" in md  # noqa: RUF001
    assert "15:10–15:19" in md  # noqa: RUF001
    assert "단일가 호가 시간(08:30–09:00, 15:20–15:30)에는 일체 주문을 제출하지 않습니다." in md  # noqa: RUF001

    # Check that in open window, HY2 sell precedes HY2I buy
    open_idx = md.index("09:05–09:15:")  # noqa: RUF001
    close_idx = md.index("15:10–15:19:")  # noqa: RUF001
    open_chunk = md[open_idx:close_idx]
    assert open_chunk.index("HY2(0193T0)") < open_chunk.index("HY2I(0197X0)")
    assert "매도" in open_chunk[: open_chunk.index("HY2I(0197X0)")]

    # Check that in close window, HY2I sell precedes HY2 buy
    close_chunk = md[close_idx:]
    assert close_chunk.index("HY2I(0197X0)") < close_chunk.index("HY2(0193T0)")
    assert "매도" in close_chunk[: close_chunk.index("HY2(0193T0)")]


def test_final_short_keeps_inverse_through_close() -> None:
    """Final short card omits close window order line and states inverse held to close."""
    settings = _fixture_settings()
    standing = Standing(None, None, None, None, None, None, None)
    card = SegmentCard(
        session=date(2026, 10, 7),
        execution_session=date(2026, 10, 8),
        final_session=True,
        action=SegmentAction.SWITCH_SHORT,
        intraday_side=IntradaySide.SHORT,
        vote_changes=(("HY2", 1.0), ("SEMI2", 1.0), ("K2", -1.0)),
        down_votes=1,
        vote_min_down=2,
        sessions_remaining_after_execution=0,
        endgame=True,
        standing=standing,
        warnings=("FINAL_SESSION", "ENDGAME_REVIEW"),
    )
    md = render_segment_markdown(card, settings)
    assert "15:10–15:19" not in md  # noqa: RUF001
    assert "HY2I(0197X0)" in md
    assert "종가까지 보유" in md


def test_hold_card_has_no_order_lines() -> None:
    """HOLD_LONG, NO_DATA, and CONTEST_OVER cards have no order lines with buy/sell outside holdings/mismatch."""
    settings = _fixture_settings()
    standing = Standing(None, None, None, None, None, None, None)

    for action, side in [
        (SegmentAction.HOLD_LONG, IntradaySide.LONG),
        (SegmentAction.NO_DATA, None),
        (SegmentAction.CONTEST_OVER, None),
    ]:
        card = SegmentCard(
            session=date(2026, 10, 7),
            execution_session=date(2026, 10, 8) if action != SegmentAction.CONTEST_OVER else None,
            final_session=False,
            action=action,
            intraday_side=side,
            vote_changes=(("HY2", -1.0), ("SEMI2", -1.0), ("K2", -1.0)),
            down_votes=3 if action != SegmentAction.CONTEST_OVER else 0,
            vote_min_down=2,
            sessions_remaining_after_execution=10 if action != SegmentAction.CONTEST_OVER else 0,
            endgame=False,
            standing=standing,
            warnings=(),
        )
        md = render_segment_markdown(card, settings)
        lines = md.splitlines()
        for line in lines:
            if "불일치 규칙" in line or "보유 점검" in line or "수동 폴백 지침" in line:
                continue
            assert "매수" not in line, f"Found 매수 in line: {line}"
            assert "매도" not in line, f"Found 매도 in line: {line}"


def test_mismatch_rule_present() -> None:
    """Non-CONTEST_OVER cards include the 08:50 holdings check and mismatch rule before orders."""
    settings = _fixture_settings()
    standing = Standing(None, None, None, None, None, None, None)
    card = SegmentCard(
        session=date(2026, 10, 7),
        execution_session=date(2026, 10, 8),
        final_session=False,
        action=SegmentAction.SWITCH_SHORT,
        intraday_side=IntradaySide.SHORT,
        vote_changes=(("HY2", 1.0), ("SEMI2", 1.0), ("K2", -1.0)),
        down_votes=1,
        vote_min_down=2,
        sessions_remaining_after_execution=10,
        endgame=False,
        standing=standing,
        warnings=(),
    )
    md = render_segment_markdown(card, settings)
    assert "08:50 보유 점검" in md
    assert "불일치 규칙" in md
    assert md.index("08:50 보유 점검") < md.index("09:05–09:15:")  # noqa: RUF001


def test_standing_rendered() -> None:
    """Standing fields render with labels and fallback to '없음' when missing."""
    settings = _fixture_settings()
    standing = Standing(
        snapshot_date=date(2026, 10, 7),
        our_equity=1.125,
        our_source=EquitySource.USER,
        our_as_of=date(2026, 10, 7),
        prize_line_equity=1.50,
        leader_equity=1.80,
        required_multiple=1.333,
    )
    card = SegmentCard(
        session=date(2026, 10, 7),
        execution_session=date(2026, 10, 8),
        final_session=False,
        action=SegmentAction.HOLD_LONG,
        intraday_side=IntradaySide.LONG,
        vote_changes=(("HY2", -1.0), ("SEMI2", -1.0), ("K2", -1.0)),
        down_votes=3,
        vote_min_down=2,
        sessions_remaining_after_execution=15,
        endgame=False,
        standing=standing,
        warnings=(),
    )
    md = render_segment_markdown(card, settings)
    assert "+12.50% (앱 입력)" in md
    assert "+50.00%" in md
    assert "+80.00%" in md
    assert "x1.33" in md
    assert "잔여 세션: 15" in md

    # Empty standing
    empty_card = SegmentCard(
        session=date(2026, 10, 7),
        execution_session=None,
        final_session=False,
        action=SegmentAction.CONTEST_OVER,
        intraday_side=None,
        vote_changes=(("HY2", None), ("SEMI2", None), ("K2", None)),
        down_votes=0,
        vote_min_down=2,
        sessions_remaining_after_execution=0,
        endgame=False,
        standing=Standing(None, None, None, None, None, None, None),
        warnings=(),
    )
    md_empty = render_segment_markdown(empty_card, settings)
    assert "- 우리 누적수익률: 없음" in md_empty
    assert "- 1위 수익률: 없음" in md_empty
    assert "- 상금권 진입 필요 배수: 없음" in md_empty


def test_card_dict_round_trip_primitives() -> None:
    """segment_to_dict produces clean JSON primitives without custom encoders."""
    standing = Standing(
        snapshot_date=date(2026, 10, 7),
        our_equity=1.1,
        our_source=EquitySource.LEDGER,
        our_as_of=date(2026, 10, 7),
        prize_line_equity=1.3,
        leader_equity=1.6,
        required_multiple=1.18,
    )
    card = SegmentCard(
        session=date(2026, 10, 7),
        execution_session=date(2026, 10, 8),
        final_session=False,
        action=SegmentAction.SWITCH_SHORT,
        intraday_side=IntradaySide.SHORT,
        vote_changes=(("HY2", -1.5), ("SEMI2", 0.5), ("K2", 1.2)),
        down_votes=1,
        vote_min_down=2,
        sessions_remaining_after_execution=10,
        endgame=False,
        standing=standing,
        warnings=("QUOTES_MISSING",),
    )
    d = segment_to_dict(card)
    raw_json = json.dumps(d)
    restored = json.loads(raw_json)
    assert restored["session"] == "2026-10-07"
    assert restored["execution_session"] == "2026-10-08"
    assert restored["action"] == "SWITCH_SHORT"
    assert restored["intraday_side"] == "SHORT"
    assert restored["standing"]["our_source"] == "LEDGER"
    assert restored["vote_changes"][0] == {"alias": "HY2", "change_pct": -1.5}


# --- Module Budget Invariant ---


def test_module_budget() -> None:
    """src/contest/segment.py must have <= 400 statements and no imports of src.cli."""
    module_path = Path("src/contest/segment.py")
    tree = ast.parse(module_path.read_text("utf-8"))
    statements = sum(1 for node in ast.walk(tree) if isinstance(node, ast.stmt))
    assert statements <= 400, f"src/contest/segment.py holds {statements} statements (budget <= 400)"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("src.cli"), f"illegal import {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("src.cli"), f"illegal import from {node.module}"


def test_non_finite_competitor_blanks_prize_line() -> None:
    """A NaN competitor return cannot be ranked, so the prize line and leader stay unknown."""
    snapshot = LeaderboardSnapshot(
        base_date=date(2026, 10, 7),
        requested_at="",
        purchases={},
        entries=(
            LeaderboardEntry(1, "a", 80.0, 1.0),
            LeaderboardEntry(2, "b", float("nan"), 1.0),
            LeaderboardEntry(3, "c", 60.0, 1.0),
            LeaderboardEntry(4, "d", 50.0, 1.0),
        ),
    )
    ledger = LedgerState(date(2026, 10, 7), 0.9, EquitySource.USER)
    standing, _ = compute_standing(snapshot, date(2026, 10, 7), "me", ledger, 3)
    assert standing.prize_line_equity is None
    assert standing.leader_equity is None
    assert standing.required_multiple is None
