"""SCENARIO-08-18 SCENARIO-B2-07"""
import argparse
import json
import tempfile
from pathlib import Path

from src.cli import cmd_decide
from src.reporting.dashboard import DailyDecision, write_decision_artifact


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


def test_SCENARIO_B2_07_cmd_decide_and_artifact(capsys) -> None:  # noqa: N802
    # SCENARIO-B2-07: cmd_decide with synthetic/temp panel exits 0; stdout contains PORTFOLIO and WHY; WHY line includes 'state='; write_decision_artifact creates JSON with keys as_of, selected (list), and each selected item has reason non-empty
    import io
    import sys

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        rc = cmd_decide(argparse.Namespace(date="2026-10-07"))
    finally:
        sys.stdout = old_stdout
    assert rc == 0
    txt = captured.getvalue()
    assert "PORTFOLIO" in txt
    assert "WHY" in txt
    # check state= present
    assert "state=" in txt
    # also test write_decision_artifact directly
    from datetime import date as _date

    dec = DailyDecision(decision_date=_date(2026, 10, 7), weights={"069500": 0.5, "451060": 0.3}, rationales={"069500": "WHY: 069500 weight=0.500 state=HOLD", "451060": "WHY: 451060 weight=0.300 state=TRIM"})
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "artifact.json"
        out_path = write_decision_artifact(dec, p)
        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert "as_of" in data
        assert "selected" in data
        assert isinstance(data["selected"], list)
        for item in data["selected"]:
            assert "reason" in item
            assert isinstance(item["reason"], str)  # noqa: PT018
            assert len(item["reason"].strip()) > 0  # noqa: PT018
            assert "state=" in item["reason"] or "WHY" in item["reason"]


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-18", "SCENARIO-B2-07"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str, capsys) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-18":
        test_SCENARIO_08_18_cmd_decide(capsys)
    if scenario_id == "SCENARIO-B2-07":
        test_SCENARIO_B2_07_cmd_decide_and_artifact(capsys)
