"""P4 CLI decomposition: features.py moved verbatim from _impl.py."""
import argparse

from src.cli.commands.features import cmd_features


def test_p4_features_missing_args_returns_1() -> None:
    assert cmd_features(argparse.Namespace()) == 1


def test_p4_features_start_after_end_returns_1() -> None:
    args = argparse.Namespace(start="2026-08-27", end="2026-01-02")
    assert cmd_features(args) == 1
