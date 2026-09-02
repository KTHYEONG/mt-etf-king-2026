# ruff: noqa
from __future__ import annotations
import argparse
from src.cli._impl import cmd_config_check as _impl, cmd_calendar as _cal
def cmd_config_check(args: argparse.Namespace) -> int:
    return _impl(args)
def cmd_calendar(args: argparse.Namespace) -> int:
    return _cal(args)
