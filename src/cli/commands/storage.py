# ruff: noqa
from __future__ import annotations
import argparse
from src.cli._impl import cmd_storage_migrate as _impl
def cmd_storage_migrate(args: argparse.Namespace) -> int:
    return _impl(args)
