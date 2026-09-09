"""CLI entry point (P4 decomposition of src/cli/_impl.py)."""

from __future__ import annotations

import argparse
import contextlib
import logging
from collections.abc import Callable, Sequence

from src.cli.commands.backtest import cmd_backtest
from src.cli.commands.champion_research import cmd_champion_research
from src.cli.commands.config import cmd_calendar, cmd_config_check
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
from src.core.logging_setup import configure_logging

logger = logging.getLogger(__name__)

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
            return int(handler(args))
        return int(func(args))
    except Exception as exc:
        logger.error(f"[SYS] main status=fail error={exc!r}")
        return 1


__all__ = ["SUBCOMMANDS", "build_parser", "main"]
