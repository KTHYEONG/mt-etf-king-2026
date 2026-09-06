# ruff: noqa
from __future__ import annotations

import argparse
import logging
from datetime import date

from src.core.calendar import get_calendar
from src.core.paths import DataPaths
from src.core.settings import get_settings
from src.data.silver import SilverBuilder

logger = logging.getLogger(__name__)


def cmd_ingest(args: argparse.Namespace) -> int:
    try:
        dataset = getattr(args, "dataset", None)
        start_s = getattr(args, "start", None)
        end_s = getattr(args, "end", None)
        dry_run = bool(getattr(args, "dry_run", False))
        if dataset is None or start_s is None or end_s is None:
            logger.error("[SYS] ingest status=fail error=missing required arguments")
            return 1
        try:
            start = date.fromisoformat(start_s) if isinstance(start_s, str) else start_s
            end = date.fromisoformat(end_s) if isinstance(end_s, str) else end_s
        except Exception as exc:
            logger.error(f"[SYS] ingest status=fail error={exc!r}")
            return 1
        settings = get_settings()
        paths = DataPaths(root=settings.data_root)
        cal = get_calendar()
        from src.data.backfill import BackfillPlanner
        from src.data.bronze import BronzeStore

        store = BronzeStore(paths)
        from src.data.providers.ratelimit import QuotaLedger, RateLimiter

        ledger = QuotaLedger(paths.state("krx_quota"), daily_quota=settings.daily_call_quota)
        limiter = RateLimiter(requests_per_second=settings.requests_per_second)
        planner = BackfillPlanner(cal, store, ledger)
        plan = planner.plan(dataset, start, end)
        if dry_run:
            logger.info(
                f"[SYS] ingest dataset={dataset} start={start} end={end} "
                f"scheduled={len(plan.scheduled)} deferred={len(plan.deferred)} already_present={plan.already_present} dry_run=True"
            )
            logger.info(
                f"[DATA] ingest plan dataset={dataset} scheduled={len(plan.scheduled)} deferred={len(plan.deferred)} already_present={plan.already_present}"
            )
            return 0
        # Real run
        import asyncio

        import httpx

        from src.data.backfill import run_backfill
        from src.data.providers.krx import KRXOpenAPIProvider

        base_url = settings.krx_base_url

        async def _run() -> int:
            async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
                provider = KRXOpenAPIProvider(client=client, base_url=base_url, limiter=limiter)
                result = await run_backfill(plan, provider, store, ledger, max_concurrency=settings.max_concurrency)
                logger.info(
                    f"[DATA] ingest written={result.written} skipped={result.skipped} failed={len(result.failed)} quota_exhausted={result.quota_exhausted}"
                )
                logger.info(
                    f"[SYS] ingest dataset={dataset} start={start} end={end} written={result.written} quota_exhausted={result.quota_exhausted}"
                )
                return 0

        return asyncio.run(_run())
    except Exception as exc:
        logger.error(f"[SYS] ingest status=fail error={exc!r}")
        return 1


def cmd_normalize(args: argparse.Namespace) -> int:
    try:
        dataset = getattr(args, "dataset", None) or "etf_daily"
        mode = getattr(args, "mode", None) or "incremental"
        if mode not in ("full", "incremental"):
            logger.error(f"[SYS] normalize status=fail error=invalid mode {mode!r}")
            return 1
        settings = get_settings()
        paths = DataPaths(root=settings.data_root)
        cal = get_calendar()
        from src.data.bronze import BronzeStore
        from src.data.validation import PanelValidator

        validator = PanelValidator(calendar=cal)
        store = BronzeStore(paths)
        builder = SilverBuilder(store, paths, validator)
        result = builder.build(dataset, mode=mode)  # type: ignore[arg-type]
        logger.info(f"[DATA] normalize dataset={dataset} mode={mode} rows={result.rows} sessions={result.sessions} path={result.path}")
        return 0
    except Exception as exc:
        logger.error(f"[SYS] normalize status=fail error={exc!r}")
        logger.error(f"[DATA] normalize status=fail dataset={getattr(args, 'dataset', 'unknown')} error={exc!r}")
        return 1
