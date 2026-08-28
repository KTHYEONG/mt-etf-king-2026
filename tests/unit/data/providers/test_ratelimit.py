from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.data.providers.ratelimit import QuotaLedger, RateLimiter


@pytest.mark.asyncio
async def test_scenario_02_05_ratelimiter_and_quotaledger(tmp_path: Path) -> None:
    """SCENARIO-02-05."""
    # RateLimiter with injected clock
    t = [0.0]

    def clock() -> float:
        return t[0]

    sleeps: list[float] = []

    async def sleeper(d: float) -> None:
        sleeps.append(d)
        t[0] += d

    rl = RateLimiter(requests_per_second=5.0, clock=clock, sleeper=sleeper)
    for _ in range(10):
        await rl.acquire()
    total = sum(sleeps)
    assert total >= 1.8 - 1e-9
    assert all(s >= -1e-12 for s in sleeps)

    # QuotaLedger
    path = tmp_path / "state" / "krx_quota.json"
    fixed = date(2026, 8, 27)

    def today() -> date:
        return fixed

    ledger = QuotaLedger(path, daily_quota=100, today=today)
    assert ledger.remaining() == 100
    ledger.consume(3)
    assert ledger.remaining() == 97

    # persists across reconstruction
    ledger2 = QuotaLedger(path, daily_quota=100, today=today)
    assert ledger2.remaining() == 97

    # resets next date
    def today_next() -> date:
        return date(2026, 8, 28)

    ledger3 = QuotaLedger(path, daily_quota=100, today=today_next)
    assert ledger3.remaining() == 100
