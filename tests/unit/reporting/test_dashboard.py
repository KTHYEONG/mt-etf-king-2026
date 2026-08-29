"""SCENARIO-08-14"""
from datetime import date

from src.reporting.dashboard import DailyDecision, build_rationale, render_dashboard


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


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-14"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-14":
        test_SCENARIO_08_14_rationale_and_dashboard()
