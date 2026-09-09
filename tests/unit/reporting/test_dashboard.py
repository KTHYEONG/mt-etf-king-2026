"""SCENARIO-08-14"""
from datetime import date

from src.reporting.dashboard import DailyDecision, build_rationale, render_dashboard, write_decision_artifact


def test_SCENARIO_08_14_rationale_and_dashboard() -> None:  # noqa: N802
    pos = {"ticker": "069500", "weight": 0.6, "state": "HOLD", "theme": "ThemeA"}
    r = build_rationale(pos)
    assert isinstance(r, str)  # noqa: PT018
    assert len(r) > 0  # noqa: PT018
    # render dashboard includes WHY and tickers
    dd = DailyDecision(
        decision_date=date(2026, 10, 7),
        weights={"069500": 0.6, "451060": 0.4},
        rationales={"069500": r, "451060": build_rationale({"ticker": "451060", "weight": 0.4})},
    )
    out = render_dashboard(dd)
    assert "WHY" in out
    assert "069500" in out
    assert "451060" in out
    # missing rationale -> omit position
    dd2 = DailyDecision(
        decision_date=date(2026, 10, 7),
        weights={"069500": 0.6, "X": 0.4},
        rationales={"069500": r, "X": ""},
    )
    out2 = render_dashboard(dd2)
    assert "X: 0.4" not in out2  # noqa: PT018


def test_write_decision_artifact_includes_order_estimate_shares_and_krw(tmp_path) -> None:
    import json
    from types import SimpleNamespace

    dd = DailyDecision(
        decision_date=date(2026, 8, 27),
        weights={"412570": 1.0},
        rationales={"412570": "WHY: 412570 weight=1.000 state=HOLD"},
    )
    est = SimpleNamespace(est_shares=1_376, est_krw=1_376 * 726_500.0)
    out_path = tmp_path / "decision.json"

    write_decision_artifact(dd, out_path, order_estimates={"412570": est})

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    item = next(s for s in payload["selected"] if s["ticker"] == "412570")
    assert item["est_shares"] == 1_376
    assert item["est_krw"] == 1_376 * 726_500.0


def test_write_decision_artifact_omits_estimate_fields_when_none_given(tmp_path) -> None:
    import json

    dd = DailyDecision(
        decision_date=date(2026, 8, 27),
        weights={"412570": 1.0},
        rationales={"412570": "WHY: 412570 weight=1.000 state=HOLD"},
    )
    out_path = tmp_path / "decision.json"

    write_decision_artifact(dd, out_path)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    item = next(s for s in payload["selected"] if s["ticker"] == "412570")
    assert "est_shares" not in item
    assert "est_krw" not in item


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-14"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-14":
        test_SCENARIO_08_14_rationale_and_dashboard()
