"""SCENARIO-10-04"""
from src.tournament.harness import resolve_leverage_scenario
from src.universe.tournament import UNKNOWN
import pytest

def test_SCENARIO_10_04_resolve() -> None:
    assert resolve_leverage_scenario("aggressive", None) is True
    assert resolve_leverage_scenario("conservative", True) is False
    assert resolve_leverage_scenario("rules", UNKNOWN) is None
    assert resolve_leverage_scenario("rules", True) is True
    assert resolve_leverage_scenario("rules", False) is False

@pytest.mark.parametrize("scenario_id", ["SCENARIO-10-04"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:
    test_SCENARIO_10_04_resolve()
