# ruff: noqa
import pytest

from src.tournament.distribution import ReturnDistribution
from src.tournament.objective import ObjectiveGateConfig, evaluate_p15_adoption_report


def _dist(name: str, returns: list[float], *, horizon: int = 2) -> ReturnDistribution:
    return ReturnDistribution.summarise(
        name=name,
        returns=returns,
        horizon=horizon,
        thresholds=[0.30, 0.40, 0.50],
        tail_weights={0.9: 1.0},
    )


def _base_config() -> ObjectiveGateConfig:
    return ObjectiveGateConfig(
        g1_prob_threshold=0.30,
        g1_min_improvement=0.02,
        g2a_ruin_threshold=-0.25,
        g2a_max_prob=0.05,
    )


def _passing_returns() -> tuple[list[float], list[float], list[float], list[float]]:
    p15_returns = [0.45] * 5 + [0.32] * 5 + [-0.10] * 20
    b1_returns = [0.42] * 3 + [0.31] * 5 + [-0.10] * 22
    b0_returns = [0.31] * 4 + [-0.26] * 2 + [0.0] * 24
    p14_returns = [0.40] * 4 + [0.31] * 4 + [-0.10] * 22
    return p15_returns, b1_returns, b0_returns, p14_returns


@pytest.mark.slow
def test_p15_full_period_adoption_report() -> None:
    config = _base_config()
    p15_returns, b1_returns, b0_returns, p14_returns = _passing_returns()
    p15 = _dist("P15", p15_returns)
    b1 = _dist("B1", b1_returns)
    b0 = _dist("B0", b0_returns)
    p14 = _dist("P14", p14_returns)

    pass_report = evaluate_p15_adoption_report(
        p15=p15,
        b1=b1,
        b0=b0,
        p14=p14,
        config=config,
        max_positive_family_count=1,
        multi_family_rate=0.0,
        artifacts_complete=True,
        leverage_scenarios=("aggressive", "conservative"),
        era_p40_deltas={"2018_2021": 0.02, "2022_2026": 0.01},
        era_b1_tail_event_counts={"2018_2021": 6, "2022_2026": 5},
    )
    assert pass_report.status == "PASS"
    assert pass_report.failures == ()
    assert pass_report.objective is not None
    assert pass_report.objective.status == "PASS"

    fail_report = evaluate_p15_adoption_report(
        p15=_dist("P15", [0.45] * 5 + [-0.30] * 25),
        b1=b1,
        b0=b0,
        p14=p14,
        config=config,
        max_positive_family_count=1,
        multi_family_rate=0.0,
        artifacts_complete=True,
        leverage_scenarios=("aggressive", "conservative"),
    )
    assert fail_report.status == "FAIL"
    assert "G2A_RUIN" in fail_report.failures

    insufficient_report = evaluate_p15_adoption_report(
        p15=_dist("P15", [0.1], horizon=10),
        b1=b1,
        b0=b0,
        p14=p14,
        config=config,
        max_positive_family_count=1,
        multi_family_rate=0.0,
        artifacts_complete=False,
        leverage_scenarios=("aggressive",),
    )
    assert insufficient_report.status == "INSUFFICIENT_EVIDENCE"
    assert insufficient_report.status != "PASS"
