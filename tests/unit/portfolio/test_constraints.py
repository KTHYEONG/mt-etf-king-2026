"""SCENARIO-08-05 SCENARIO-08-15 SCENARIO-09-04"""
from src.portfolio.constraints import apply_gross_exposure_cap, apply_liquidity_cap, gross_exposure, rebalance_band


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


def test_SCENARIO_09_04_gross_exposure_cap() -> None:  # noqa: N802
    assert gross_exposure({"A": 0.8}, {"A": 2}) == 1.6
    capped = apply_gross_exposure_cap({"A": 0.9}, {"A": 2}, max_gross=1.60)
    assert gross_exposure(capped, {"A": 2}) <= 1.60 + 1e-9


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-05", "SCENARIO-08-15", "SCENARIO-09-04"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-05":
        test_SCENARIO_08_05_liquidity_cap()
    if scenario_id == "SCENARIO-08-15":
        test_SCENARIO_08_15_rebalance_band()
    if scenario_id == "SCENARIO-09-04":
        test_SCENARIO_09_04_gross_exposure_cap()
