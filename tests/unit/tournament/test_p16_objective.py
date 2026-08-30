from pathlib import Path
from src.tournament.distribution import ReturnDistribution
from src.tournament.objective import ObjectiveGateConfig, evaluate_p16_adoption_report

def _dist(name: str, returns: list[float], horizon: int = 2) -> ReturnDistribution:
    return ReturnDistribution.summarise(name=name, returns=returns, horizon=horizon, thresholds=[0.30, 0.40, 0.50], tail_weights={0.9: 1.0})

def test_evaluate_p16_adoption_report_pass_and_fail() -> None:
    config = ObjectiveGateConfig.from_yaml(Path("configs/gates.yaml"))
    p16_ok = [0.55] * 6 + [0.42] * 4 + [0.32] * 4 + [-0.10] * 16
    b1 = [0.42] * 3 + [0.31] * 5 + [-0.10] * 22
    b0 = [0.31] * 4 + [-0.26] * 2 + [0.0] * 24
    p14 = [0.40] * 4 + [0.31] * 4 + [-0.10] * 22
    passed = evaluate_p16_adoption_report(p16=_dist("P16", p16_ok), b1=_dist("B1", b1), b0=_dist("B0", b0), p14=_dist("P14", p14), config=config, artifacts_complete=True, leverage_scenarios=("aggressive", "conservative"), skip_capacity_violations=0, vehicle_mult2_rate=0.9)
    assert passed.status == "PASS"
    assert passed.failures == ()
    p16_low40 = [0.32] * 8 + [0.10] * 22
    failed = evaluate_p16_adoption_report(p16=_dist("P16", p16_low40), b1=_dist("B1", b1), b0=_dist("B0", b0), p14=_dist("P14", p14), config=config, artifacts_complete=True, leverage_scenarios=("aggressive", "conservative"), skip_capacity_violations=0, vehicle_mult2_rate=0.9)
    assert failed.status == "FAIL"
    assert "P40_VS_B1" in failed.failures or "P50_VS_B1" in failed.failures
    cap = evaluate_p16_adoption_report(p16=_dist("P16", p16_ok), b1=_dist("B1", b1), b0=_dist("B0", b0), p14=_dist("P14", p14), config=config, artifacts_complete=True, leverage_scenarios=("aggressive", "conservative"), skip_capacity_violations=1, vehicle_mult2_rate=0.9)
    assert cap.status == "FAIL"
    assert "CAPACITY_ON_CONVEXITY" in cap.failures
    ruin = evaluate_p16_adoption_report(p16=_dist("P16", [0.45] * 5 + [-0.30] * 25), b1=_dist("B1", b1), b0=_dist("B0", b0), p14=_dist("P14", p14), config=config, artifacts_complete=True, leverage_scenarios=("aggressive", "conservative"), skip_capacity_violations=0, vehicle_mult2_rate=0.9)
    assert ruin.status == "FAIL"
    assert "G2A_RUIN" in ruin.failures
    missing = evaluate_p16_adoption_report(p16=_dist("P16", p16_ok), b1=_dist("B1", b1), b0=_dist("B0", b0), p14=_dist("P14", p14), config=config, artifacts_complete=False, leverage_scenarios=("aggressive", "conservative"), skip_capacity_violations=0, vehicle_mult2_rate=0.9)
    assert missing.status != "PASS"
    assert "MISSING_ARTIFACT" in missing.failures
    tiny = evaluate_p16_adoption_report(p16=_dist("P16", [0.1], horizon=10), b1=_dist("B1", b1), b0=_dist("B0", b0), p14=_dist("P14", p14), config=config, artifacts_complete=True, leverage_scenarios=("aggressive", "conservative"), skip_capacity_violations=0, vehicle_mult2_rate=0.9)
    assert tiny.status == "INSUFFICIENT_EVIDENCE"
