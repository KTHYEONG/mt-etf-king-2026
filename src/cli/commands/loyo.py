# ruff: noqa
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.core.paths import DataPaths
from src.core.settings import get_settings

logger = logging.getLogger(__name__)


def cmd_loyo(args: argparse.Namespace) -> int:
    try:
        run_id = getattr(args, "run_id", None)
        if run_id is None:
            run_id = getattr(args, "runId", None)
        inc_id = getattr(args, "incumbent_run_id", None)
        if inc_id is None:
            inc_id = getattr(args, "incumbentRunId", None)
        if not run_id:
            logger.error("[SYS] loyo status=fail error=missing --run-id")
            return 1
        from src.tournament.loyo import evaluate_promotion_robustness, write_loyo_report

        settings = get_settings()
        paths = DataPaths(root=settings.data_root)
        try:
            run_dir = paths.results(str(run_id))
        except Exception:
            run_dir = Path("docs/results") / str(run_id)
        windows_path = Path(run_dir) / "windows.parquet"
        if not windows_path.exists():
            logger.error(f"[SYS] loyo status=fail error=missing windows for run {run_id}")
            return 1
        import polars as pl

        try:
            windows = pl.read_parquet(str(windows_path))
        except Exception as exc:
            logger.error(f"[SYS] loyo status=fail error=read windows {exc!r}")
            return 1
        incumbent_windows = None
        if inc_id:
            try:
                inc_dir = paths.results(str(inc_id))
            except Exception:
                inc_dir = Path("docs/results") / str(inc_id)
            inc_path = Path(inc_dir) / "windows.parquet"
            if not inc_path.exists():
                logger.error(f"[SYS] loyo status=fail error=missing windows for incumbent {inc_id}")
                return 1
            try:
                incumbent_windows = pl.read_parquet(str(inc_path))
            except Exception as exc:
                logger.error(f"[SYS] loyo status=fail error=read incumbent windows {exc!r}")
                return 1
        result = evaluate_promotion_robustness(
            candidate_windows=windows, incumbent_windows=incumbent_windows
        )
        out = write_loyo_report(Path(run_dir), result)
        logger.info(
            f"[SYS] loyo run={run_id} status={result.status} "
            f"concentration={result.concentration_2025_2026:.4f} path={out}"
        )
        return 0
    except Exception as exc:
        logger.error(f"[SYS] loyo status=fail error={exc!r}")
        return 1
