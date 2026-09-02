# ruff: noqa
from __future__ import annotations
import argparse
from src.cli._impl import cmd_replay as _impl
def cmd_replay(args: argparse.Namespace) -> int:
    return _impl(args)
