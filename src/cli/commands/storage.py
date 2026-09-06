# ruff: noqa
from __future__ import annotations

import argparse
import logging

from src.core.paths import DataPaths
from src.core.settings import get_settings

logger = logging.getLogger(__name__)


def cmd_storage_migrate(args: argparse.Namespace) -> int:
    try:
        settings = get_settings()
        paths = DataPaths(root=settings.data_root)
        from src.data.bronze import BronzeStore

        store = BronzeStore(paths)
        # wiring: ensure BronzeStore.migrate_plain_to_gzip referenced
        endpoints = ["etp/etf_bydd_trd"]
        # allow override via args.endpoint if provided
        ep_arg = getattr(args, "endpoint", None)
        if ep_arg:
            endpoints = [str(ep_arg)]
        # also handle common alias etf_bydd_trd -> etp/etf_bydd_trd?
        total = {"migrated": 0, "skipped_existing_gz": 0, "failed": 0, "deleted_plain": 0}
        delete_plain = bool(getattr(args, "delete_plain", True))
        # if --no-delete supplied?
        if getattr(args, "no_delete", False):
            delete_plain = False
        for ep in endpoints:
            res = store.migrate_plain_to_gzip(ep, delete_plain=delete_plain)
            for k in total:
                total[k] += int(res.get(k, 0))
        from src.reporting.results import rebuild_runs_registry
        rebuilt = rebuild_runs_registry(paths)
        logger.info(
            f"[DATA] storage-migrate endpoints={endpoints} migrated={total['migrated']} "
            f"skipped_existing_gz={total['skipped_existing_gz']} failed={total['failed']} "
            f"deleted_plain={total['deleted_plain']} results_rebuilt={rebuilt}"
        )
        logger.info(
            f"[SYS] storage-migrate migrated={total['migrated']} skipped={total['skipped_existing_gz']} failed={total['failed']} deleted_plain={total['deleted_plain']}"
        )
        return 1 if total["failed"] > 0 else 0
    except Exception as exc:
        logger.error(f"[SYS] storage-migrate status=fail error={exc!r}")
        return 1
