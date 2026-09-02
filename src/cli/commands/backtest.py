# ruff: noqa
from __future__ import annotations

import argparse

from src.cli.dispatch import normalize_cli_model_arg
from src.strategies.registry import resolve_strategy_id


def cmd_backtest(args: argparse.Namespace) -> int:
    normalize_cli_model_arg(args)
    resolve_strategy_id(str(getattr(args, "model", "")))
    from src.cli._impl import cmd_backtest as _impl

    return _impl(args)
