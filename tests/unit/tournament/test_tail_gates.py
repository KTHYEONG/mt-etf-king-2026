"""SCENARIO-10-08"""
from src.tournament.distribution import evaluate_tail_gates
import pytest

def test_SCENARIO_10_08_gates() -> None:
    # PASS case: p40 >= b1+0.02, p50 >= b1+0.01, cvar >= b1-0.05, activity >=0.25
    status, fails = evaluate_tail_gates(0.12, 0.10, 0.095, 0.083, -0.35, -0.386, 0.30)
    assert status == "PASS"
    assert fails == []
    # fail p40 only
    status, fails = evaluate_tail_gates(0.11, 0.10, 0.095, 0.083, -0.35, -0.386, 0.30)
    assert status == "FAIL"
    assert "p_gt_40" in fails
    # fail p50 only
    status, fails = evaluate_tail_gates(0.12, 0.10, 0.09, 0.09, -0.35, -0.386, 0.30)
    assert status == "FAIL"
    assert "p_gt_50" in fails
    # fail cvar only
    status, fails = evaluate_tail_gates(0.12, 0.10, 0.095, 0.083, -0.44, -0.386, 0.30)
    assert status == "FAIL"
    assert "cvar_05" in fails
    # fail activity only
    status, fails = evaluate_tail_gates(0.12, 0.10, 0.095, 0.083, -0.35, -0.386, 0.20)
    assert status == "FAIL"
    assert "vehicle_activity" in fails

@pytest.mark.parametrize("scenario_id", ["SCENARIO-10-08"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:
    test_SCENARIO_10_08_gates()
