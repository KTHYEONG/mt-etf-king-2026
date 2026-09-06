"""P4 CLI decomposition: data.py (ingest/normalize) moved verbatim from _impl.py."""
import argparse

from src.cli.commands.data import cmd_ingest, cmd_normalize


def test_p4_data_ingest_missing_args_returns_1() -> None:
    assert cmd_ingest(argparse.Namespace()) == 1


def test_p4_data_normalize_invalid_mode_returns_1() -> None:
    assert cmd_normalize(argparse.Namespace(dataset="etf_daily", mode="bogus")) == 1
