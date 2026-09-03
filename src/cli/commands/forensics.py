# ruff: noqa
from __future__ import annotations

import argparse


def cmd_forensics(args: argparse.Namespace) -> int:  # thin re-export to _impl.cmd_forensics
    from src.cli._impl import cmd_forensics as _impl

    return _impl(args)
