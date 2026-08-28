from __future__ import annotations

from datetime import date

import pytest

from src.core.calendar import TradingCalendar


def test_session_count_tournament_periods() -> None:
    """SCENARIO-01-01: 2026 대회 구간 세션 수 검증."""
    cal = TradingCalendar()
    assert cal.session_count(date(2026, 9, 21), date(2026, 11, 13)) == 36
    assert cal.session_count(date(2025, 9, 22), date(2025, 11, 14)) == 35


def test_is_session_and_sessions_range() -> None:
    """SCENARIO-01-02: is_session 및 sessions 로직."""
    cal = TradingCalendar()
    assert cal.is_session(date(2026, 8, 15)) is False
    assert cal.is_session(date(2026, 8, 17)) is False
    assert cal.is_session(date(2026, 8, 27)) is True
    assert cal.sessions(date(2026, 8, 13), date(2026, 8, 20)) == [
        date(2026, 8, 13),
        date(2026, 8, 14),
        date(2026, 8, 18),
        date(2026, 8, 19),
        date(2026, 8, 20),
    ]


def test_previous_next_session_and_validation() -> None:
    """SCENARIO-01-03: previous/next session 순환성 및 오류 처리."""
    cal = TradingCalendar()
    assert cal.previous_session(date(2026, 8, 18)) == date(2026, 8, 14)
    # round-trip: next(previous(d)) == d for all d in range
    for d in cal.sessions(date(2026, 1, 5), date(2026, 8, 27)):
        assert cal.next_session(cal.previous_session(d)) == d
    # session_count with start > end raises
    with pytest.raises(ValueError, match=r"start .* > end"):
        cal.session_count(date(2026, 8, 20), date(2026, 8, 13))
    # previous_session accepted non-session input (e.g., Sunday)
    assert cal.previous_session(date(2026, 8, 16)) == date(2026, 8, 14)
    assert cal.next_session(date(2026, 8, 16)) == date(2026, 8, 18)
