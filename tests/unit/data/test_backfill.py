from __future__ import annotations

from datetime import date, datetime, UTC
from pathlib import Path

import pytest

from src.core.calendar import TradingCalendar
from src.core.paths import DataPaths
from src.data.backfill import BackfillPlanner, run_backfill
from src.data.bronze import BronzeRecord, BronzeStore
from src.data.providers.base import TransientProviderError
from src.data.providers.ratelimit import QuotaLedger


def test_scenario_02_07_planner_sessions_and_already_present(tmp_path: Path) -> None:
    """SCENARIO-02-07."""
    paths = DataPaths(root=tmp_path)
    store = BronzeStore(paths)

    def today() -> date:
        return date(2026, 8, 27)

    ledger = QuotaLedger(paths.state("krx_quota"), daily_quota=8000, today=today)
    cal = TradingCalendar(name="XKRX")
    planner = BackfillPlanner(cal, store, ledger, data_start=date(2010, 1, 4))

    plan = planner.plan("etf_daily", date(2026, 8, 13), date(2026, 8, 20))
    assert plan.scheduled == (
        date(2026, 8, 13),
        date(2026, 8, 14),
        date(2026, 8, 18),
        date(2026, 8, 19),
        date(2026, 8, 20),
    )

    # after store holds 2026-08-14
    rec = BronzeRecord(
        endpoint="etp/etf_bydd_trd",
        bas_dd=date(2026, 8, 14),
        fetched_at=datetime.now(UTC),
        http_status=200,
        row_count=0,
        rows=[],
    )
    store.write(rec)
    plan2 = planner.plan("etf_daily", date(2026, 8, 13), date(2026, 8, 20))
    assert len(plan2.scheduled) == 4
    assert plan2.already_present == 1

    # data_start cutoff
    plan3 = planner.plan("etf_daily", date(2009, 1, 1), date(2010, 1, 10))
    for d in plan3.scheduled:
        assert d >= date(2010, 1, 4)


@pytest.mark.asyncio
async def test_scenario_02_08_quota_split_and_resume(tmp_path: Path) -> None:
    """SCENARIO-02-08."""
    paths = DataPaths(root=tmp_path)
    store = BronzeStore(paths)

    def today() -> date:
        return date(2026, 8, 27)

    ledger = QuotaLedger(paths.state("krx_quota"), daily_quota=3, today=today)
    cal = TradingCalendar(name="XKRX")
    planner = BackfillPlanner(cal, store, ledger, data_start=date(2010, 1, 4))

    plan = planner.plan("etf_daily", date(2026, 8, 13), date(2026, 8, 20))
    assert len(plan.scheduled) == 3
    assert len(plan.deferred) == 2

    class StubProvider:
        def __init__(self) -> None:
            self.calls: list[date] = []

        async def fetch_session(self, endpoint: str, bas_dd: date):  # type: ignore[override]
            self.calls.append(bas_dd)
            return [{"BAS_DD": bas_dd.strftime("%Y%m%d")}]

    stub = StubProvider()
    result = await run_backfill(plan, stub, store, ledger, max_concurrency=2)
    assert result.written == 3
    assert result.quota_exhausted is True
    assert len(store.available_sessions("etp/etf_bydd_trd")) == 3

    # second plan after ledger date rolls over
    def today_next() -> date:
        return date(2026, 8, 28)

    ledger2 = QuotaLedger(paths.state("krx_quota"), daily_quota=3, today=today_next)
    planner2 = BackfillPlanner(cal, store, ledger2, data_start=date(2010, 1, 4))
    plan2 = planner2.plan("etf_daily", date(2026, 8, 13), date(2026, 8, 20))
    assert len(plan2.scheduled) == 2
    assert plan2.already_present == 3

    result2 = await run_backfill(plan2, stub, store, ledger2, max_concurrency=2)
    assert result2.written == 2
    assert len(store.available_sessions("etp/etf_bydd_trd")) == 5


@pytest.mark.asyncio
async def test_scenario_02_09_failure_partial(tmp_path: Path) -> None:
    """SCENARIO-02-09."""
    paths = DataPaths(root=tmp_path)
    store = BronzeStore(paths)

    def today() -> date:
        return date(2026, 8, 27)

    ledger = QuotaLedger(paths.state("krx_quota"), daily_quota=100, today=today)
    cal = TradingCalendar(name="XKRX")
    planner = BackfillPlanner(cal, store, ledger)
    plan = planner.plan("etf_daily", date(2026, 8, 13), date(2026, 8, 20))
    assert len(plan.scheduled) == 5

    class FailingStub:
        async def fetch_session(self, endpoint: str, bas_dd: date):  # type: ignore[override]
            if bas_dd == date(2026, 8, 18):
                raise TransientProviderError("fail")
            return [{"x": "y"}]

    stub = FailingStub()
    result = await run_backfill(plan, stub, store, ledger, max_concurrency=2)
    assert result.written == 4
    assert len(result.failed) == 1
    assert date(2026, 8, 18) in result.failed
    assert result.quota_exhausted is False

    planner2 = BackfillPlanner(cal, store, ledger)
    plan2 = planner2.plan("etf_daily", date(2026, 8, 13), date(2026, 8, 20))
    assert plan2.scheduled == (date(2026, 8, 18),)
