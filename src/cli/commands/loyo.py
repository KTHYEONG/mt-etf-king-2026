# ruff: noqa
from __future__ import annotations

import argparse


def cmd_loyo(args: argparse.Namespace) -> int:  # thin re-export to _impl.cmd_loyo
    from src.cli._impl import cmd_loyo as _impl

    return _impl(args)
