"""SCENARIO-08-09 SCENARIO-B2-03 SCENARIO-B2-04"""
from src.portfolio.policy import PortfolioPolicy
from src.portfolio.sizing import ConfidenceSizingConfig


def test_SCENARIO_08_09_portfolio_policy_path_dependent() -> None:  # noqa: N802
    assert PortfolioPolicy.path_dependent is True
    p = PortfolioPolicy(sizing_config=ConfidenceSizingConfig())
    assert p.path_dependent is True
    dec = p.allocate({"A": 0.5, "B": 0.3, "C": 0.2})
    assert hasattr(dec, "weights")
    assert sum(dec.weights.values()) <= 1.0 + 1e-9
    # empty scores -> empty weights
    dec2 = p.allocate({})
    assert dec2.weights == {}


def test_SCENARIO_B2_03_breakdown_and_reset() -> None:  # noqa: N802
    # SCENARIO-B2-03: After allocate with theme_states forcing BREAKDOWN on top ticker, that ticker weight==0.0 and rationale contains 'state=EXIT' or 'state=WATCH'; reset_trackers clears
    p = PortfolioPolicy(sizing_config=ConfidenceSizingConfig())
    scores = {"AAA": 0.10, "BBB": 0.05, "CCC": 0.02}
    # force BREAKDOWN on AAA
    dec = p.allocate(scores, theme_states={"AAA": "BREAKDOWN"})
    assert dec.weights.get("AAA", 1.0) == 0.0
    assert "AAA" in dec.rationale  # type: ignore
    r = dec.rationale["AAA"]  # type: ignore
    assert "state=EXIT" in r or "state=WATCH" in r
    # reset should clear trackers
    p.reset_trackers()
    # next allocate should start HOLD - allocate without breakdown should give positive weight for AAA if top
    dec2 = p.allocate(scores)
    # AAA should now have positive weight (since HOLD)
    assert dec2.weights.get("AAA", 0) > 0


def test_SCENARIO_B2_04_cooldown_reenter() -> None:  # noqa: N802
    # SCENARIO-B2-04: Sequence EXIT then WATCH for cooldown-1 sessions then RECOVERY at >=3 yields RE_ENTER then HOLD
    p = PortfolioPolicy(sizing_config=ConfidenceSizingConfig())
    scores = {"TICK": 0.08, "OTHER": 0.01}
    # Step1: BREAKDOWN -> EXIT
    dec1 = p.allocate(scores, theme_states={"TICK": "BREAKDOWN"})
    assert dec1.weights["TICK"] == 0.0
    assert "state=EXIT" in dec1.rationale["TICK"]  # type: ignore
    # Step2: WATCH for cooldown-1 sessions (2 sessions) with non-RECOVERY
    for _ in range(2):
        dec = p.allocate(scores, theme_states={"TICK": "OVERHEATED"})
        assert dec.weights["TICK"] == 0.0
        # should be WATCH
        assert "state=WATCH" in dec.rationale["TICK"]  # type: ignore
    # Step3: RECOVERY at sessions_since_exit>=3 -> RE_ENTER
    dec3 = p.allocate(scores, theme_states={"TICK": "RECOVERY"})
    # RE_ENTER should have positive weight (m=1)
    assert dec3.weights["TICK"] > 0
    assert "state=RE_ENTER" in dec3.rationale["TICK"]  # type: ignore
    # Next session after RE_ENTER should be HOLD with positive weight
    dec4 = p.allocate(scores, theme_states={"TICK": "LEADING"})
    assert dec4.weights["TICK"] > 0
    assert "state=HOLD" in dec4.rationale["TICK"]  # type: ignore


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-09", "SCENARIO-B2-03", "SCENARIO-B2-04"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-09":
        test_SCENARIO_08_09_portfolio_policy_path_dependent()
    if scenario_id == "SCENARIO-B2-03":
        test_SCENARIO_B2_03_breakdown_and_reset()
    if scenario_id == "SCENARIO-B2-04":
        test_SCENARIO_B2_04_cooldown_reenter()
