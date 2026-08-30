from src.tournament.distribution import ReturnDistribution
from src.tournament.objective import ObjectiveGateConfig, evaluate_p15_adoption_report, evaluate_p16_adoption_report

def _dist(name: str, returns: list[float], horizon: int = 2) -> ReturnDistribution:
    return ReturnDistribution.summarise(name=name, returns=returns, horizon=horizon, thresholds=[0.30, 0.40, 0.50], tail_weights={0.9: 1.0})

def test_p16_adoption_report_vehicle_and_leverage_gates() -> None:
    config = ObjectiveGateConfig(g1_prob_threshold=0.30, g1_min_improvement=0.02, g2a_ruin_threshold=-0.25, g2a_max_prob=0.05)
    p16_ok = [0.55] * 6 + [0.42] * 4 + [0.32] * 4 + [-0.10] * 16
    b1 = [0.42] * 3 + [0.31] * 5 + [-0.10] * 22
    b0 = [0.31] * 4 + [-0.26] * 2 + [0.0] * 24
    p14 = [0.40] * 4 + [0.31] * 4 + [-0.10] * 22
    veh = evaluate_p16_adoption_report(p16=_dist("P16", p16_ok), b1=_dist("B1", b1), b0=_dist("B0", b0), p14=_dist("P14", p14), config=config, artifacts_complete=True, leverage_scenarios=("aggressive", "conservative"), skip_capacity_violations=0, vehicle_mult2_rate=0.0)
    assert veh.status == "FAIL"
    assert "VEHICLE_ACTIVITY" in veh.failures
    lev = evaluate_p16_adoption_report(p16=_dist("P16", p16_ok), b1=_dist("B1", b1), b0=_dist("B0", b0), p14=_dist("P14", p14), config=config, artifacts_complete=True, leverage_scenarios=("aggressive",), skip_capacity_violations=0, vehicle_mult2_rate=0.9)
    assert lev.status == "FAIL"
    assert "LEVERAGE_SCENARIOS" in lev.failures
    ok = evaluate_p16_adoption_report(p16=_dist("P16", p16_ok), b1=_dist("B1", b1), b0=_dist("B0", b0), p14=_dist("P14", p14), config=config, artifacts_complete=True, leverage_scenarios=("aggressive", "conservative"), skip_capacity_violations=0, vehicle_mult2_rate=0.9)
    assert ok.status == "PASS"
    assert callable(evaluate_p15_adoption_report)
