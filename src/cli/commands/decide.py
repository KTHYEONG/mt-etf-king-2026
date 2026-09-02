# ruff: noqa
from __future__ import annotations

import argparse

from src.cli.dispatch import normalize_cli_model_arg
from src.strategies.registry import resolve_strategy_id


def cmd_decide(args: argparse.Namespace) -> int:
    _model_arg = getattr(args, "model", None)
    if _model_arg is not None:
        resolve_strategy_id(str(_model_arg))
        normalize_cli_model_arg(args)
    from src.cli._impl import cmd_decide as _impl

    return _impl(args)
