from __future__ import annotations

from datetime import date

import pytest

from src.core.calendar import HolidayOverride, TradingCalendar, kst_today, load_holiday_overrides
from src.core.config import ConfigError
from src.data.validation import find_future_dates


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


def test_SCENARIO_03A_01_kst_today_and_find_future_dates() -> None:  # noqa: N802
    """SCENARIO-03A-01"""
    today = kst_today()
    assert isinstance(today, date)
    assert find_future_dates([date(2026, 8, 27), date(2026, 8, 29), date(2026, 8, 28)], date(2026, 8, 28)) == [date(2026, 8, 29)]
    assert find_future_dates([], date(2026, 8, 28)) == []


def test_session_count_matches_sessions_without_allocating_list() -> None:
    import inspect
    from datetime import date
    from src.core.calendar import TradingCalendar
    cal = TradingCalendar()
    for s, e in ((date(2018, 1, 2), date(2026, 8, 27)), (date(2026, 9, 21), date(2026, 11, 13)), (date(2026, 8, 27), date(2026, 8, 27))):
        assert cal.session_count(s, e) == len(cal.sessions(s, e))
    src = inspect.getsource(TradingCalendar.session_count)
    assert "self.sessions(" not in src


globals()["test SCENARIO-03A-01"] = test_SCENARIO_03A_01_kst_today_and_find_future_dates  # noqa: E402, F401, N816


def test_configured_closures_are_not_sessions() -> None:
    """Declared KRX closures are removed from the effective session set."""
    cal = TradingCalendar()
    assert cal.is_session(date(2026, 6, 3)) is False
    assert cal.is_session(date(2026, 7, 17)) is False
    june_july = cal.sessions(date(2026, 6, 1), date(2026, 7, 31))
    assert date(2026, 6, 3) not in june_july
    assert date(2026, 7, 17) not in june_july


def test_neighbors_skip_the_closure() -> None:
    """Session navigation jumps over a declared closure in both directions."""
    cal = TradingCalendar()
    assert cal.previous_session(date(2026, 6, 4)) == date(2026, 6, 2)
    assert cal.next_session(date(2026, 7, 16)) == date(2026, 7, 20)


def test_counts_agree_with_lists_across_overrides() -> None:
    """session_count matches len(sessions) on ranges spanning each override day."""
    cal = TradingCalendar()
    for start, end in (
        (date(2026, 6, 1), date(2026, 6, 5)),
        (date(2026, 7, 15), date(2026, 7, 21)),
        (date(2026, 6, 1), date(2026, 7, 31)),
    ):
        assert cal.session_count(start, end) == len(cal.sessions(start, end))


def test_explicit_empty_overrides_expose_upstream_defect() -> None:
    """With overrides disabled the upstream library still lists the closure as a session."""
    raw = TradingCalendar(holiday_overrides=())
    assert raw.is_session(date(2026, 6, 3)) is True


def test_redundant_override_is_noop(caplog: pytest.LogCaptureFixture) -> None:
    """An override on a non-session day changes nothing and is logged at DEBUG."""
    raw = TradingCalendar(holiday_overrides=())
    assert TradingCalendar(holiday_overrides=[date(2026, 8, 15)]).sessions(
        date(2026, 8, 1), date(2026, 8, 31)
    ) == raw.sessions(date(2026, 8, 1), date(2026, 8, 31))
    with caplog.at_level("DEBUG", logger="src.core.calendar"):
        TradingCalendar(holiday_overrides=[date(2026, 8, 15)])
    assert any("override redundant day=2026-08-15" in record.message for record in caplog.records)


def test_load_holiday_overrides_sorted() -> None:
    """The configured XKRX overrides load sorted by day with reasons kept."""
    overrides = load_holiday_overrides("XKRX")
    assert [override.day for override in overrides] == [date(2026, 6, 3), date(2026, 7, 17)]
    assert all(isinstance(override, HolidayOverride) and override.reason for override in overrides)


def test_load_holiday_overrides_unknown_exchange() -> None:
    """An exchange with no config entry yields no overrides."""
    assert load_holiday_overrides("XNYS") == ()


def test_load_holiday_overrides_absent_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config without the override key yields no overrides."""
    monkeypatch.setattr("src.core.calendar.load_config", lambda name: {"calendar": {"exchange": "XKRX"}})
    assert load_holiday_overrides("XKRX") == ()


@pytest.mark.parametrize(
    "entries",
    [
        [{"date": "2026-06-03"}],
        [{"date": "2026/06/03", "reason": "slash date"}],
        [{"date": "20260603", "reason": "basic format"}],
        [{"date": "2026-02-30", "reason": "no such day"}],
        [{"date": "2026-06-03", "reason": "   "}],
        [{"date": "2026-06-03", "reason": "first"}, {"date": "2026-06-03", "reason": "second"}],
        ["not-a-mapping"],
        [{"date": "2026-06-03", "reason": "ok"}, "not-a-mapping"],
    ],
)
def test_load_holiday_overrides_malformed(
    monkeypatch: pytest.MonkeyPatch, entries: list[object]
) -> None:
    """Malformed override entries fail closed with ConfigError."""
    monkeypatch.setattr(
        "src.core.calendar.load_config",
        lambda name: {"calendar": {"holiday_overrides": {"XKRX": entries}}},
    )
    with pytest.raises(ConfigError):
        load_holiday_overrides("XKRX")


@pytest.mark.parametrize(
    "config",
    [
        {"calendar": "XKRX"},
        {"calendar": {"holiday_overrides": ["2026-06-03"]}},
        {"calendar": {"holiday_overrides": {"XKRX": {"date": "2026-06-03"}}}},
    ],
)
def test_load_holiday_overrides_malformed_shape(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, object]
) -> None:
    """Malformed override containers fail closed with ConfigError."""
    monkeypatch.setattr("src.core.calendar.load_config", lambda name: config)
    with pytest.raises(ConfigError):
        load_holiday_overrides("XKRX")
