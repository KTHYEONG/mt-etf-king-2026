# ruff: noqa
from pathlib import Path

from src.tournament.distribution import ReturnDistribution
from src.tournament.objective import (
    ObjectiveGateConfig,
    evaluate_objective_gates,
    evaluate_p15_adoption_report,
    paired_tail_delta_ci,
)


def test_objective_gate_enforces_configured_ruin_probability() -> None:
    candidate_returns = [0.35] * 20 + [-0.30] * 10
    baseline_returns = [0.35] * 10 + [0.0] * 20
    horizon = 2
    cand = ReturnDistribution.summarise(name="cand", returns=candidate_returns, horizon=horizon, thresholds=[0.30], tail_weights={0.9: 1.0})
    base = ReturnDistribution.summarise(name="B0", returns=baseline_returns, horizon=horizon, thresholds=[0.30], tail_weights={0.9: 1.0})
    config = ObjectiveGateConfig(g1_prob_threshold=0.30, g1_min_improvement=0.02, g2a_ruin_threshold=-0.25, g2a_max_prob=0.05)
    result = evaluate_objective_gates(cand, base, config)
    assert result.status == "FAIL"
    assert "G2A_RUIN" in result.failures
    cfg2 = ObjectiveGateConfig.from_yaml(Path("configs/gates.yaml"))
    assert cfg2.g1_prob_threshold == 0.30
    assert cfg2.g2a_max_prob == 0.05


def test_objective_gate_rejects_insufficient_sample_or_artifacts() -> None:
    config = ObjectiveGateConfig(g1_prob_threshold=0.30, g1_min_improvement=0.02, g2a_ruin_threshold=-0.25, g2a_max_prob=0.05)
    cand = ReturnDistribution.summarise(name="cand", returns=[0.1], horizon=10, thresholds=[0.30], tail_weights={0.9: 1.0})
    base = ReturnDistribution.summarise(name="B0", returns=[0.1], horizon=2, thresholds=[0.30], tail_weights={0.9: 1.0})
    assert cand.n_effective < 1
    res = evaluate_objective_gates(cand, base, config)
    assert res.status == "INSUFFICIENT_EVIDENCE"
    assert res.status != "PASS"
    cand2 = ReturnDistribution.summarise(name="cand", returns=[], horizon=2, thresholds=[0.30], tail_weights={0.9: 1.0})
    base2 = ReturnDistribution.summarise(name="B0", returns=[0.1, 0.2], horizon=2, thresholds=[0.30], tail_weights={0.9: 1.0})
    res2 = evaluate_objective_gates(cand2, base2, config)
    assert res2.status == "INSUFFICIENT_EVIDENCE"
    assert res2.status != "PASS"
    ci = paired_tail_delta_ci([0.4, 0.5, 0.6], [0.1, 0.2, 0.3], threshold=0.30, expected_block=2, n_resamples=100, seed=0)
    assert isinstance(ci, tuple) and len(ci) == 2


def test_p15_adoption_report_pass_and_fail_paths() -> None:
    config = ObjectiveGateConfig(g1_prob_threshold=0.30, g1_min_improvement=0.02, g2a_ruin_threshold=-0.25, g2a_max_prob=0.05)
    horizon = 2
    p15_returns = [0.45] * 5 + [0.32] * 5 + [-0.10] * 20
    b1_returns = [0.42] * 3 + [0.31] * 5 + [-0.10] * 22
    b0_returns = [0.31] * 4 + [-0.26] * 2 + [0.0] * 24
    p14_returns = [0.40] * 4 + [0.31] * 4 + [-0.10] * 22
    p15 = ReturnDistribution.summarise(name="P15", returns=p15_returns, horizon=horizon, thresholds=[0.30, 0.40], tail_weights={0.9: 1.0})
    b1 = ReturnDistribution.summarise(name="B1", returns=b1_returns, horizon=horizon, thresholds=[0.30, 0.40], tail_weights={0.9: 1.0})
    b0 = ReturnDistribution.summarise(name="B0", returns=b0_returns, horizon=horizon, thresholds=[0.30, 0.40], tail_weights={0.9: 1.0})
    p14 = ReturnDistribution.summarise(name="P14", returns=p14_returns, horizon=horizon, thresholds=[0.30, 0.40], tail_weights={0.9: 1.0})
    passed = evaluate_p15_adoption_report(
        p15=p15,
        b1=b1,
        b0=b0,
        p14=p14,
        config=config,
        max_positive_family_count=1,
        multi_family_rate=0.0,
        artifacts_complete=True,
        leverage_scenarios=("aggressive", "conservative"),
    )
    assert passed.status == "PASS"
    failed = evaluate_p15_adoption_report(
        p15=ReturnDistribution.summarise(name="P15", returns=[0.45] * 5 + [-0.30] * 25, horizon=horizon, thresholds=[0.30, 0.40], tail_weights={0.9: 1.0}),
        b1=b1,
        b0=b0,
        p14=p14,
        config=config,
        max_positive_family_count=1,
        multi_family_rate=0.0,
        artifacts_complete=True,
        leverage_scenarios=("aggressive", "conservative"),
    )
    assert failed.status == "FAIL"
    assert "G2A_RUIN" in failed.failures
