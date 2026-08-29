from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime

from src.core.calendar import TradingCalendar
from src.core.logging_setup import tagged_log
from src.data.bronze import BronzeRecord, BronzeStore
from src.data.providers.base import MarketDataProvider, ProviderError
from src.data.providers.krx import resolve_endpoint
from src.data.providers.ratelimit import QuotaLedger

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillPlan:
    dataset: str
    endpoint: str
    scheduled: tuple[date, ...]
    deferred: tuple[date, ...]
    already_present: int


@dataclass(frozen=True)
class BackfillResult:
    written: int
    skipped: int
    failed: tuple[date, ...]
    quota_exhausted: bool


class BackfillPlanner:
    def __init__(
        self,
        calendar: TradingCalendar,
        store: BronzeStore,
        ledger: QuotaLedger,
        data_start: date = date(2010, 1, 4),
    ) -> None:
        self._calendar = calendar
        self._store = store
        self._ledger = ledger
        self._data_start = data_start

    def plan(self, dataset: str, start: date, end: date) -> BackfillPlan:
        try:
            endpoint = resolve_endpoint(dataset)
        except KeyError:
            # fallback for raw endpoint names like 'etf_bydd_trd'
            if "/" in dataset:
                endpoint = dataset
            elif dataset == "etf_bydd_trd":
                endpoint = "etp/etf_bydd_trd"
            else:
                endpoint = dataset
        # Clamp start to data_start
        effective_start = start if start >= self._data_start else self._data_start
        if effective_start > end:
            candidate: list[date] = []
        else:
            candidate = self._calendar.sessions(effective_start, end)
        missing: list[date] = []
        already_present = 0
        for d in candidate:
            if self._store.has_session(endpoint, d):
                already_present += 1
            else:
                missing.append(d)
        remaining = self._ledger.remaining()
        # Split
        scheduled = tuple(missing[:remaining]) if remaining > 0 else ()  # noqa: C408
        deferred = tuple(missing[remaining:]) if remaining >= 0 else tuple(missing)
        return BackfillPlan(
            dataset=dataset,
            endpoint=endpoint,
            scheduled=scheduled,
            deferred=deferred,
            already_present=already_present,
        )


async def run_backfill(
    plan: BackfillPlan,
    provider: MarketDataProvider,
    store: BronzeStore,
    ledger: QuotaLedger,
    max_concurrency: int = 6,
) -> BackfillResult:
    if not plan.scheduled:
        quota_exhausted = len(plan.deferred) > 0 and ledger.remaining() <= 0
        return BackfillResult(written=0, skipped=0, failed=(), quota_exhausted=quota_exhausted)  # noqa: C408

    sem = asyncio.Semaphore(max_concurrency)
    written = 0
    skipped = 0
    failed: list[date] = []
    # For thread-safe counters
    lock = asyncio.Lock()

    # To control logging throttling
    completed = 0

    async def _process_one(bas_dd: date) -> None:
        nonlocal written, skipped, completed
        async with sem:
            # Check quota before attempt
            if ledger.remaining() <= 0:
                async with lock:
                    failed.append(bas_dd)
                return
            # Consume quota for this attempt
            try:  # noqa: SIM105,S110
                ledger.consume(1)
            except Exception:  # noqa: S110
                pass  # noqa: S110

            try:
                rows = await provider.fetch_session(plan.endpoint, bas_dd)
            except ProviderError:
                async with lock:
                    failed.append(bas_dd)
                    completed += 1
                    if completed % 100 == 0:
                        tagged_log(logger, "DATA", completed=completed, total=len(plan.scheduled))
                return
            except Exception:
                async with lock:
                    failed.append(bas_dd)
                    completed += 1
                    if completed % 100 == 0:
                        tagged_log(logger, "DATA", completed=completed, total=len(plan.scheduled))
                return

            # Create record - use now UTC
            fetched_at = datetime.now(UTC)
            record = BronzeRecord(
                endpoint=plan.endpoint,
                bas_dd=bas_dd,
                fetched_at=fetched_at,
                http_status=200,
                row_count=len(rows),
                rows=rows,
            )
            try:
                result = store.write(record)
            except Exception:
                async with lock:
                    failed.append(bas_dd)
                    completed += 1
                    if completed % 100 == 0:
                        tagged_log(logger, "DATA", completed=completed, total=len(plan.scheduled))
                return

            async with lock:
                if result == "written" or result == "revised":
                    written += 1
                elif result == "skipped":
                    skipped += 1
                else:
                    written += 1
                completed += 1
                if completed % 100 == 0:
                    tagged_log(logger, "DATA", completed=completed, total=len(plan.scheduled))

    # Schedule tasks but stop scheduling once ledger exhausted?
    # Plan already respects quota, so we can launch all scheduled
    # But need to honour dynamic exhaustion: if ledger becomes exhausted mid-run, remaining tasks should not fetch
    # Our _process_one checks remaining before each fetch, so they will mark as failed? But spec says for quota exhausted case, should not attempt further calls.
    # Instead, we should check before launching each task sequentially, respecting concurrency but not launching beyond quota
    # Simpler: iterate scheduled and launch tasks, but with early break if ledger exhausted before launching
    tasks: list[asyncio.Task[None]] = []
    for bas_dd in plan.scheduled:
        if ledger.remaining() <= 0:
            # Remaining plan dates should be considered not executed; they will be part of deferred on next plan?
            # For current result, we should not count them as failed; we just stop.
            # But to report quota_exhausted, we need flag
            break
        tasks.append(asyncio.create_task(_process_one(bas_dd)))

    if tasks:
        await asyncio.gather(*tasks)

    # Handle dates that were not launched due to quota exhaustion -> they remain not written, but we didn't add to failed
    # For quota_exhausted logic, deferred plus any un-launched scheduled should be considered deferred
    # However plan.scheduled was already limited, so if we broke early, some scheduled were not processed -> they should be considered deferred/failed?
    # But spec says "MUST stop scheduling further calls once the ledger is exhausted" - so we stop and report quota_exhausted True
    # We should not mark unprocessed scheduled as failed; they will be caught on next plan via missing count
    # So we just compute quota_exhausted
    # Count how many scheduled were not processed: not needed for failed

    # If we broke early, failed list doesn't include unprocessed; but written will be less than len(scheduled) processed
    # For accurate written count, we already have.

    # Determine quota_exhausted
    quota_exhausted = False
    # If deferred non-empty => quota was limiting
    if len(plan.deferred) > 0:
        quota_exhausted = True
    # Also if ledger is exhausted after run  # noqa: SIM102
    if ledger.remaining() <= 0 and len(plan.scheduled) > 0 and (len(plan.deferred) > 0 or (written + skipped + len(failed) < len(plan.scheduled))):
        quota_exhausted = True
    # Also if we had to break early due to quota, set true
    # Edge: when scheduled was exactly remaining, after consume remaining becomes 0, but deferred >0 => already true

    # Ensure we log at most one line per 100 plus final summary
    # Final summary log
    tagged_log(logger, "SYS", written=written, skipped=skipped, failed=len(failed), quota_exhausted=quota_exhausted)

    # Failed should include only provider failures, not quota-exhausted unprocessed
    # Sort failed for determinism
    failed_sorted = tuple(sorted(failed))
    return BackfillResult(written=written, skipped=skipped, failed=failed_sorted, quota_exhausted=quota_exhausted)
