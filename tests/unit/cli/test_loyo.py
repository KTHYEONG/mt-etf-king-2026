"""P4 CLI decomposition: loyo.py moved verbatim from _impl.py."""
import argparse

from src.cli.commands.loyo import cmd_loyo


def test_p4_loyo_missing_run_id_returns_1() -> None:
    assert cmd_loyo(argparse.Namespace()) == 1
