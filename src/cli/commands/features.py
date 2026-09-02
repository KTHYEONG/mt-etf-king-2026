# ruff: noqa
from __future__ import annotations
import argparse
from src.cli._impl import cmd_features as _impl
def cmd_features(args: argparse.Namespace) -> int:
    return _impl(args)
