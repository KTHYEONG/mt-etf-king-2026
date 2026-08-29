"""SCENARIO-08-18"""
import argparse

from src.cli import cmd_decide


def test_SCENARIO_08_18_cmd_decide(capsys) -> None:  # noqa: N802
    args = argparse.Namespace(date="2026-10-07")
    rc = cmd_decide(args)
    assert rc == 0
    out = capsys.readouterr().out + capsys.readouterr().err
    # also capture via capsys after call, but need to capture print output
    # Re-run with capture
    import io
    import sys

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        cmd_decide(argparse.Namespace(date="2026-10-07"))
    finally:
        sys.stdout = old_stdout
    txt = captured.getvalue()
    assert "PORTFOLIO" in txt
    assert "WHY" in txt


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-18"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str, capsys) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-18":
        test_SCENARIO_08_18_cmd_decide(capsys)
