"""CLI entry point (P4 decomposition of src/cli/_impl.py)."""

from __future__ import annotations

import argparse
import contextlib
import logging
from collections.abc import Callable, Sequence

from src.cli.commands.backtest import cmd_backtest
from src.cli.commands.champion_research import cmd_champion_research
from src.cli.commands.config import cmd_calendar, cmd_config_check
from src.cli.commands.contest import (
    cmd_contest_archive,
    cmd_contest_daily,
    cmd_contest_seed_reference,
    cmd_contest_weekly,
)
from src.cli.commands.data import cmd_ingest, cmd_normalize
from src.cli.commands.decide import cmd_decide
from src.cli.commands.feasibility_audit import cmd_feasibility_audit
from src.cli.commands.features import cmd_features
from src.cli.commands.forensics import cmd_forensics
from src.cli.commands.loyo import cmd_loyo
from src.cli.commands.pipeline import cmd_daily_refresh
from src.cli.commands.replay import cmd_replay
from src.cli.commands.storage import cmd_storage_migrate
from src.cli.commands.universe import cmd_universe
from src.cli.parser import build_parser
from src.core.config import config_value
from src.core.locks import DataLockTimeout, data_lock
from src.core.logging_setup import configure_logging
from src.core.paths import DataPaths
from src.core.settings import get_settings

logger = logging.getLogger(__name__)

_EXCLUSIVE_DATA_COMMANDS = frozenset({"ingest", "normalize", "features", "daily-refresh", "storage-migrate"})
_SHARED_DATA_COMMANDS = frozenset({"contest-weekly", "contest-daily"})

SUBCOMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "config-check": cmd_config_check,
    "calendar": cmd_calendar,
    "ingest": cmd_ingest,
    "normalize": cmd_normalize,
    "universe": cmd_universe,
    "features": cmd_features,
    "backtest": cmd_backtest,
    "forensics": cmd_forensics,
    "loyo": cmd_loyo,
    "replay": cmd_replay,
    "decide": cmd_decide,
    "daily-refresh": cmd_daily_refresh,
    "storage-migrate": cmd_storage_migrate,
    "champion-research": cmd_champion_research,
    "feasibility-audit": cmd_feasibility_audit,
    "contest-archive": cmd_contest_archive,
    "contest-seed-reference": cmd_contest_seed_reference,
    "contest-weekly": cmd_contest_weekly,
    "contest-daily": cmd_contest_daily,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # Argparse exits with 2 on unknown subcommand; convert to return code
        return int(exc.code) if isinstance(exc.code, int) else 1
    # configure logging after parse_args (INV-LOG-BOOT)
    try:
        lvl = getattr(args, "log_level", "INFO")
        configure_logging(level=str(lvl))
    except Exception:
        with contextlib.suppress(Exception):
            configure_logging()
    # wiring anchor explicitly
    configure_logging()
    # Unknown subcommand or no subcommand
    if not hasattr(args, "func"):
        parser.print_usage()
        return 2
    func = getattr(args, "func", None)
    if func is None:
        return 2
    try:
        # Also support registry lookup fallback
        sub_name = getattr(args, "subcommand", None)
        if sub_name is not None and sub_name in SUBCOMMANDS:
            handler = SUBCOMMANDS[args.subcommand]
            if sub_name in _EXCLUSIVE_DATA_COMMANDS or sub_name in _SHARED_DATA_COMMANDS:
                timeout_s = float(config_value("base", "pipeline", "lock_timeout_s", required=True))
                lock_path = DataPaths(root=get_settings().data_root).lock("pipeline")
                try:
                    with data_lock(
                        lock_path, shared=sub_name in _SHARED_DATA_COMMANDS, timeout_s=timeout_s
                    ):
                        return int(handler(args))
                except DataLockTimeout as exc:
                    logger.error(f"[SYS] data_lock status=fail reason=timeout error={exc!r}")
                    return 1
            return int(handler(args))
        return int(func(args))
    except Exception as exc:
        logger.error(f"[SYS] main status=fail error={exc!r}")
        return 1


__all__ = ["SUBCOMMANDS", "build_parser", "main"]
