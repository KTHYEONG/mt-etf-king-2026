"""P4 CLI decomposition: forensics.py moved verbatim from _impl.py."""
import argparse

from src.cli.commands.forensics import cmd_forensics


def test_p4_forensics_missing_run_id_returns_1() -> None:
    assert cmd_forensics(argparse.Namespace()) == 1
