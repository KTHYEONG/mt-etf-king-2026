"""SCENARIO-08-09"""
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


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-09"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-09":
        test_SCENARIO_08_09_portfolio_policy_path_dependent()
