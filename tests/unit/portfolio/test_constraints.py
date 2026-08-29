"""SCENARIO-08-05 SCENARIO-08-15"""
from src.portfolio.constraints import apply_liquidity_cap, rebalance_band


def test_SCENARIO_08_05_liquidity_cap() -> None:  # noqa: N802
    out = apply_liquidity_cap({"T": 0.90}, {"T": 1e10}, capital=1e9, participation=0.01)
    assert out["T"] <= 0.10 + 1e-6


def test_SCENARIO_08_15_rebalance_band() -> None:  # noqa: N802
    out = rebalance_band(target={"A": 0.55}, current={"A": 0.52}, min_delta=0.05)
    assert abs(out["A"] - 0.52) < 1e-9
    # also test that large delta triggers trade
    out2 = rebalance_band(target={"A": 0.60}, current={"A": 0.52}, min_delta=0.05)
    assert abs(out2["A"] - 0.60) < 1e-9


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-05", "SCENARIO-08-15"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-05":
        test_SCENARIO_08_05_liquidity_cap()
    if scenario_id == "SCENARIO-08-15":
        test_SCENARIO_08_15_rebalance_band()
