from __future__ import annotations

import pytest

from src.tournament.distribution import evaluate_adoption_gates


def test_evaluate_adoption_gates_pass_and_fail() -> None:
    status, fails = evaluate_adoption_gates(0.12, 0.117, 0.115, 0.095, -0.34, -0.368, 0.30)
    assert status == "PASS"
    assert fails == []
    # p_gt_30 below B1 -> FAIL with p_gt_30
    status2, fails2 = evaluate_adoption_gates(0.10, 0.117, 0.115, 0.095, -0.34, -0.368, 0.30)
    assert status2 == "FAIL"
    assert "p_gt_30" in fails2
    # p_gt_40 below B1+2pp -> FAIL
    status3, fails3 = evaluate_adoption_gates(0.12, 0.117, 0.10, 0.095, -0.34, -0.368, 0.30)
    assert status3 == "FAIL"
    assert "p_gt_40" in fails3
    # cvar below B1-5pp -> FAIL
    status4, fails4 = evaluate_adoption_gates(0.12, 0.117, 0.115, 0.095, -0.45, -0.368, 0.30)
    assert status4 == "FAIL"
    assert "cvar_05" in fails4
    # vehicle below threshold -> FAIL
    status5, fails5 = evaluate_adoption_gates(0.12, 0.117, 0.115, 0.095, -0.34, -0.368, 0.20)
    assert status5 == "FAIL"
    assert "vehicle_activity" in fails5
    # custom min_vehicle_rate
    status6, fails6 = evaluate_adoption_gates(0.12, 0.117, 0.115, 0.095, -0.34, -0.368, 0.30, min_vehicle_rate=0.5)
    assert status6 == "FAIL"
    assert "vehicle_activity" in fails6


@pytest.mark.parametrize("scenario_id", ["test_evaluate_adoption_gates_pass_and_fail"])
def test_scenario_wrapper(scenario_id: str) -> None:
    test_evaluate_adoption_gates_pass_and_fail()
