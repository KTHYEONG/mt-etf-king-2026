# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from src.cli.commands.backtest import cmd_backtest
from src.cli.commands.decide import cmd_decide
from src.cli.commands.config import cmd_config_check, cmd_calendar
from src.cli.commands.data import cmd_ingest, cmd_normalize
from src.cli.commands.universe import cmd_universe
from src.cli.commands.features import cmd_features
from src.cli.commands.replay import cmd_replay
from src.cli.commands.storage import cmd_storage_migrate

SUBCOMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "config-check": cmd_config_check,
    "calendar": cmd_calendar,
    "ingest": cmd_ingest,
    "normalize": cmd_normalize,
    "universe": cmd_universe,
    "features": cmd_features,
    "backtest": cmd_backtest,
    "replay": cmd_replay,
    "decide": cmd_decide,
    "storage-migrate": cmd_storage_migrate,
}

def build_parser() -> argparse.ArgumentParser:
    _ = cmd_backtest
    from src.cli._impl import build_parser as _impl

    return _impl()

def main(argv: Sequence[str] | None = None) -> int:
    from src.cli._impl import main as _impl
    return _impl(argv)

__all__ = ["build_parser", "main", "SUBCOMMANDS"]
