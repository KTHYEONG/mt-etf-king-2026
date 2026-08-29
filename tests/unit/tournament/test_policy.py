"""SCENARIO-08-11 SCENARIO-08-12"""
from src.tournament.policy import AggressionInput, AggressionPolicy, risk_multiplier


def test_SCENARIO_08_11_aggression_disabled() -> None:  # noqa: N802
    pol = AggressionPolicy(enabled=False)
    for delta in [0.05, -0.05, 0.0]:
        inp = AggressionInput(delta=delta, n=10)
        assert pol.apply(inp) == 1.0
        assert risk_multiplier(inp, {}) != 1.0 or delta == 0.0  # ensure risk_multiplier not always 1


def test_SCENARIO_08_12_risk_multiplier_monotonic() -> None:  # noqa: N802
    assert risk_multiplier(AggressionInput(delta=0.05, n=10), {}) > risk_multiplier(AggressionInput(delta=0.01, n=10), {})
    # for delta=-0.05, n=5 < n=10 => multiplier(5) > multiplier(10)
    m5 = risk_multiplier(AggressionInput(delta=-0.05, n=5), {})
    m10 = risk_multiplier(AggressionInput(delta=-0.05, n=10), {})
    assert m5 > m10


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-11", "SCENARIO-08-12"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-11":
        test_SCENARIO_08_11_aggression_disabled()
    if scenario_id == "SCENARIO-08-12":
        test_SCENARIO_08_12_risk_multiplier_monotonic()
