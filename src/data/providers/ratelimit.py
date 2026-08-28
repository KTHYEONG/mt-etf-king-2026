from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    _KST = ZoneInfo("Asia/Seoul")
except Exception:
    _KST = None  # type: ignore[assignment]


def _kst_today() -> date:
    if _KST is not None:
        return datetime.now(_KST).date()
    return datetime.now().date()


class RateLimiter:
    def __init__(
        self,
        requests_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0")
        self._rps = requests_per_second
        self._interval = 1.0 / requests_per_second
        self._interval_ns = round(1e9 / requests_per_second)  # noqa: RUF046
        self._clock = clock
        self._sleeper = sleeper
        self._next_allowed_ns: int = round(clock() * 1e9)  # noqa: RUF046
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now_ns = round(self._clock() * 1e9)  # noqa: RUF046
            if now_ns < self._next_allowed_ns:
                sleep_for = (self._next_allowed_ns - now_ns) / 1e9
                if sleep_for > 0:
                    await self._sleeper(sleep_for)
                    now_ns = round(self._clock() * 1e9)  # noqa: RUF046
            base_ns = self._next_allowed_ns if now_ns < self._next_allowed_ns else now_ns
            self._next_allowed_ns = base_ns + self._interval_ns


class QuotaLedger:
    def __init__(
        self,
        path: Path,
        daily_quota: int,
        today: Callable[[], date] = _kst_today,
    ) -> None:
        self._path = path
        self._daily_quota = daily_quota
        self._today = today
        self._consumed: int = 0
        self._stored_date: date | None = None
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._consumed = 0
            self._stored_date = None
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            d_str = data.get("date")
            consumed = int(data.get("consumed", 0))
            if d_str:
                parsed = date.fromisoformat(d_str)
                today = self._today()
                if parsed == today:
                    self._stored_date = parsed
                    self._consumed = consumed
                else:
                    self._stored_date = None
                    self._consumed = 0
            else:
                self._consumed = 0
                self._stored_date = None
        except Exception:
            self._consumed = 0
            self._stored_date = None

    def _persist(self) -> None:
        today = self._today()
        payload = {"date": today.isoformat(), "consumed": self._consumed}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # atomic write via temp
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp.replace(self._path)
        self._stored_date = today

    def remaining(self) -> int:
        today = self._today()
        # if stored date mismatches today, quota resets
        if self._stored_date is not None and self._stored_date != today:
            # Check persisted file date vs today; but _stored_date already checked at load time
            # If today changed after init, we need to reload logic
            try:
                if self._path.exists():
                    with self._path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    d_str = data.get("date")
                    if d_str:
                        parsed = date.fromisoformat(d_str)
                        if parsed != today:
                            return self._daily_quota
                        return max(0, self._daily_quota - int(data.get("consumed", 0)))
            except Exception:
                return self._daily_quota
            return self._daily_quota
        # Also handle case where file date is old but _stored_date is None
        if self._stored_date is None and self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                d_str = data.get("date")
                if d_str:
                    parsed = date.fromisoformat(d_str)
                    if parsed != today:
                        return self._daily_quota
                    # update internal state
                    self._consumed = int(data.get("consumed", 0))
                    self._stored_date = parsed
                    return max(0, self._daily_quota - self._consumed)
            except Exception:
                return self._daily_quota
        if self._stored_date is None:
            # Check if file says today
            if self._path.exists():
                try:
                    with self._path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    d_str = data.get("date")
                    if d_str and date.fromisoformat(d_str) == today:
                        self._consumed = int(data.get("consumed", 0))
                        self._stored_date = today
                        return max(0, self._daily_quota - self._consumed)
                except Exception:  # noqa: S110
                    pass  # noqa: S110
            return self._daily_quota
        return max(0, self._daily_quota - self._consumed)

    def consume(self, n: int = 1) -> None:
        if n <= 0:
            return
        today = self._today()
        # If date changed, reset
        if self._stored_date is not None and self._stored_date != today:
            self._consumed = 0
            self._stored_date = today
        elif self._stored_date is None:
            # Need to check file
            if self._path.exists():
                try:
                    with self._path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    d_str = data.get("date")
                    if d_str:
                        parsed = date.fromisoformat(d_str)
                        if parsed == today:
                            self._consumed = int(data.get("consumed", 0))
                            self._stored_date = parsed
                        else:
                            self._consumed = 0
                            self._stored_date = today
                    else:
                        self._consumed = 0
                        self._stored_date = today
                except Exception:
                    self._consumed = 0
                    self._stored_date = today
            else:
                self._stored_date = today
                self._consumed = 0
        # Now check remaining
        remaining = self._daily_quota - self._consumed
        if remaining <= 0:
            # Still persist? But quota exhausted
            # Don't consume beyond quota? Still increment?
            # Spec: consume one quota unit per attempted call, stop scheduling when exhausted.
            # So we should still increment but cap? For remaining logic, we cap at 0.
            # We'll still increment consumed to reflect overrun? But better to allow consume even when exhausted, remaining stays 0.
            self._consumed += n
            self._persist()
            return
        self._consumed += n
        self._persist()
