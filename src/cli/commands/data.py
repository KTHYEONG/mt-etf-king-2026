# ruff: noqa
from __future__ import annotations
import argparse
from src.cli._impl import cmd_ingest as _ingest, cmd_normalize as _norm
def cmd_ingest(args: argparse.Namespace) -> int:
    return _ingest(args)
def cmd_normalize(args: argparse.Namespace) -> int:
    return _norm(args)
