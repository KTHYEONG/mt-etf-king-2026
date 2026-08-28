from __future__ import annotations

import bisect
import functools
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xcals


def kst_today() -> date:
    return datetime.now(ZoneInfo("Asia/Seoul")).date()


class TradingCalendar:
    def __init__(
        self,
        name: str = "XKRX",
        start: date = date(2009, 1, 1),
        end: date | None = None,
    ) -> None:
        self.name = name
        self._start = start
        if end is None:
            end = date.today() + timedelta(days=730)
        self._end = end
        self._cal = xcals.get_calendar(name, start=start, end=end)
        # Cache sorted date list for binary search
        sessions_idx = self._cal.sessions
        # sessions is DatetimeIndex; convert to date objects
        self._sessions: list[date] = [d.date() for d in sessions_idx.to_pydatetime()]

    def is_session(self, day: date) -> bool:
        return bool(self._cal.is_session(day))

    def sessions(self, start: date, end: date) -> list[date]:
        if start > end:
            raise ValueError(f"start {start} > end {end}")
        # Use underlying calendar range then filter inclusive
        # sessions_in_range is inclusive and efficient
        idx = self._cal.sessions_in_range(start, end)
        return [d.date() for d in idx.to_pydatetime()]

    def session_count(self, start: date, end: date) -> int:
        if start > end:
            raise ValueError(f"start {start} > end {end}")
        return len(self.sessions(start, end))

    def previous_session(self, day: date, offset: int = 1) -> date:
        if offset < 1:
            raise ValueError("offset must be >= 1")
        idx = bisect.bisect_left(self._sessions, day)
        target = idx - offset
        if target < 0 or target >= len(self._sessions):
            raise ValueError(f"previous_session offset {offset} out of range for {day}")
        return self._sessions[target]

    def next_session(self, day: date, offset: int = 1) -> date:
        if offset < 1:
            raise ValueError("offset must be >= 1")
        idx = bisect.bisect_left(self._sessions, day)
        is_session_day = idx < len(self._sessions) and self._sessions[idx] == day
        target = idx + offset if is_session_day else idx + offset - 1
        if target < 0 or target >= len(self._sessions):
            raise ValueError(f"next_session offset {offset} out of range for {day}")
        return self._sessions[target]


@functools.lru_cache(maxsize=8)
def get_calendar(name: str = "XKRX") -> TradingCalendar:
    return TradingCalendar(name=name)
