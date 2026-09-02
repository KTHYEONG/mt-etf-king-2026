# ruff: noqa
from __future__ import annotations
import argparse
from src.cli._impl import cmd_universe as _impl
def cmd_universe(args: argparse.Namespace) -> int:
    return _impl(args)
