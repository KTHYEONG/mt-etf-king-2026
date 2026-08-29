"""SCENARIO-08-07 SCENARIO-08-08"""
from src.portfolio.state import PositionState, transition_position


def test_SCENARIO_08_07_cooldown() -> None:  # noqa: N802
    assert transition_position(PositionState.EXIT, "RECOVERY", 1, 3) == PositionState.WATCH
    assert transition_position(PositionState.EXIT, "RECOVERY", 3, 3) == PositionState.RE_ENTER
    assert transition_position(PositionState.WATCH, "RECOVERY", 2, 3) == PositionState.WATCH
    assert transition_position(PositionState.WATCH, "RECOVERY", 3, 3) == PositionState.RE_ENTER


def test_SCENARIO_08_08_theme_mapping() -> None:  # noqa: N802
    assert transition_position(PositionState.HOLD, "LEADING", 0, 3) == PositionState.HOLD
    assert transition_position(PositionState.HOLD, "OVERHEATED", 0, 3) == PositionState.TRIM
    assert transition_position(PositionState.HOLD, "BREAKDOWN", 0, 3) == PositionState.EXIT


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-07", "SCENARIO-08-08"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-07":
        test_SCENARIO_08_07_cooldown()
    if scenario_id == "SCENARIO-08-08":
        test_SCENARIO_08_08_theme_mapping()
