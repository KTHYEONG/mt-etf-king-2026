from __future__ import annotations

import bisect
import functools
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

from src.core.config import ConfigError, load_config

logger = logging.getLogger(__name__)

_ISO_DAY: re.Pattern[str] = re.compile(r"\d{4}-\d{2}-\d{2}")


def kst_today() -> date:
    return datetime.now(ZoneInfo("Asia/Seoul")).date()


@dataclass(frozen=True, slots=True)
class HolidayOverride:
    """An exchange closure that the pinned `exchange_calendars` release still lists as a session.

    Attributes:
        day: Closed calendar date (exchange-local).
        reason: Human-readable evidence for the closure, kept for audit.
    """

    day: date
    reason: str


def load_holiday_overrides(exchange: str) -> tuple[HolidayOverride, ...]:
    """Load the declared closures for `exchange` from `configs/base.yaml` (`calendar.holiday_overrides`).

    The upstream calendar lags ad-hoc closures (elections, newly designated holidays). Declaring them in config keeps
    the correction reviewable and independent of the library release.

    Returns:
        Overrides sorted by day; empty when the exchange has no entry or the key is absent.

    Raises:
        ConfigError: the entry for `exchange` is not a list, an item lacks `date`/`reason`, a date is not ISO
            `YYYY-MM-DD`, a reason is empty, or a date is duplicated.
    """
    data = load_config("base")
    calendar_cfg = data.get("calendar", {})
    if not isinstance(calendar_cfg, Mapping):
        raise ConfigError("calendar.holiday_overrides must be a mapping")
    overrides_cfg = calendar_cfg.get("holiday_overrides", {})
    if not isinstance(overrides_cfg, Mapping):
        raise ConfigError("calendar.holiday_overrides must be a mapping")
    entries = overrides_cfg.get(exchange, [])
    if not isinstance(entries, list):
        raise ConfigError(f"holiday overrides for {exchange!r} must be a list")
    seen: set[date] = set()
    out: list[HolidayOverride] = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise ConfigError(f"holiday override for {exchange!r} must be a mapping")
        raw_day = item.get("date")
        reason = item.get("reason")
        if not isinstance(raw_day, str) or _ISO_DAY.fullmatch(raw_day) is None:
            raise ConfigError(f"holiday override for {exchange!r} needs an ISO date (YYYY-MM-DD)")
        try:
            day = date.fromisoformat(raw_day)
        except ValueError as exc:
            raise ConfigError(f"holiday override has an invalid date: {raw_day!r}") from exc
        if not isinstance(reason, str) or not reason.strip():
            raise ConfigError(f"holiday override for {raw_day} needs a non-empty reason")
        if day in seen:
            raise ConfigError(f"duplicate holiday override for {raw_day}")
        seen.add(day)
        out.append(HolidayOverride(day=day, reason=reason))
    out.sort(key=lambda override: override.day)
    return tuple(out)


class TradingCalendar:
    """Exchange session calendar: the upstream `exchange_calendars` sessions minus declared holiday overrides.

    Every query answers from one effective session set so that ingest planning, validation, feature grids, target
    session resolution, and simulations agree on which days traded.
    """

    def __init__(
        self,
        name: str = "XKRX",
        start: date = date(2009, 1, 1),
        end: date | None = None,
        holiday_overrides: Iterable[date] | None = None,
    ) -> None:
        """Build the effective calendar.

        Args:
            name: Exchange code understood by `exchange_calendars`.
            start: First date covered.
            end: Last date covered; defaults to today + 730 days.
            holiday_overrides: Dates to remove from the upstream session set. None loads
                `load_holiday_overrides(name)`; an explicit empty iterable yields the raw upstream calendar.

        Raises:
            ConfigError: the configured override list is malformed (only when `holiday_overrides` is None).
        """
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
        if holiday_overrides is None:
            override_days = {override.day for override in load_holiday_overrides(name)}
        else:
            override_days = set(holiday_overrides)
        self._override_days = frozenset(override_days)
        if override_days:
            upstream = set(self._sessions)
            for redundant in sorted(override_days - upstream):
                logger.debug(f"[SYS] calendar override redundant day={redundant.isoformat()}")
            dropped = override_days & upstream
            if dropped:
                self._sessions = [day for day in self._sessions if day not in dropped]

    def is_session(self, day: date) -> bool:
        idx = bisect.bisect_left(self._sessions, day)
        return idx < len(self._sessions) and self._sessions[idx] == day

    def sessions(self, start: date, end: date) -> list[date]:
        if start > end:
            raise ValueError(f"start {start} > end {end}")
        # Use underlying calendar range then filter inclusive
        # sessions_in_range is inclusive and efficient
        idx = self._cal.sessions_in_range(start, end)
        days = [d.date() for d in idx.to_pydatetime()]
        if self._override_days:
            days = [day for day in days if day not in self._override_days]
        return days

    def session_count(self, start: date, end: date) -> int:
        if start > end:
            raise ValueError(f"start {start} > end {end}")
        left = bisect.bisect_left(self._sessions, start)
        right = bisect.bisect_right(self._sessions, end)
        return right - left

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
