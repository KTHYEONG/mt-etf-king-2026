"""SCENARIO-B2-09"""
import argparse
import io
import sys

from src.cli import cmd_replay


def test_SCENARIO_B2_09_replay_algo_logs(capsys) -> None:  # noqa: N802
    """SCENARIO-B2-09: cmd_replay --model P08 --year 2025 on synthetic panel exits 0 and log/stdout contains at least 35 lines matching '[ALGO]' with 'decision_date=' and 'WHY' substring"""
    args = argparse.Namespace(model="P08", year="2025")
    # capture stdout and also logger goes to capsys?
    rc = cmd_replay(args)
    assert rc == 0
    out = capsys.readouterr().out + capsys.readouterr().err
    # also capture via StringIO redirect
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = captured_out, captured_err
    try:
        cmd_replay(argparse.Namespace(model="P08", year="2025"))
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    txt = captured_out.getvalue() + captured_err.getvalue() + out
    lines = [l for l in txt.splitlines() if "[ALGO]" in l and "decision_date=" in l and "WHY" in l]
    assert len(lines) >= 35
