"""SCENARIO-08-11 SCENARIO-08-12 SCENARIO-09-08"""
from src.portfolio.policy import PortfolioPolicy
from src.portfolio.sizing import ConfidenceSizingConfig
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


def test_SCENARIO_09_08_aggression_scales_weights() -> None:  # noqa: N802
    pol = AggressionPolicy(enabled=False)
    inp = AggressionInput(delta=0.10, n=5)
    assert pol.apply(inp) == 1.0
    enabled = AggressionPolicy(enabled=True, config={"factor": 2.0, "epsilon": 0.001})
    p = PortfolioPolicy(
        sizing_config=ConfidenceSizingConfig(),
        state_enabled=False,
        aggression=enabled,
    )
    dec = p.allocate(
        {"A": 0.50, "B": 0.30, "C": 0.20},
        aggression_input=inp,
    )
    assert sum(dec.weights.values()) <= 1.0 + 1e-9


def test_overlay_should_cash_identity_never_locks() -> None:
    from src.tournament.policy import house_money_should_cash, overlay_should_cash

    assert overlay_should_cash("identity", 0.80, 1, 0.50, 5) is False
    assert overlay_should_cash("raw", 0.80, 0, 0.50, 5) is False
    assert overlay_should_cash("none", 0.80, 5, 0.50, 5) is False
    assert overlay_should_cash("house_money", 0.80, 1, 0.50, 5) is True
    assert overlay_should_cash("late_lock", 0.80, 5, 0.50, 5) is True
    assert overlay_should_cash("house_money", 0.80, 20, 0.50, 5) is False
    assert overlay_should_cash("unknown_mode", 0.80, 1, 0.50, 5) is False
    assert overlay_should_cash("house_money", 0.80, 1, 0.50, 5) is house_money_should_cash(0.80, 1, 0.50, 5)


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-11", "SCENARIO-08-12", "SCENARIO-09-08"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-11":
        test_SCENARIO_08_11_aggression_disabled()
    if scenario_id == "SCENARIO-08-12":
        test_SCENARIO_08_12_risk_multiplier_monotonic()
    if scenario_id == "SCENARIO-09-08":
        test_SCENARIO_09_08_aggression_scales_weights()
