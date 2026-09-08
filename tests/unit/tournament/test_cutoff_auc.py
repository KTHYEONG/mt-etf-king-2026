# ruff: noqa
def test_smooth_cutoff_utility_clamps_and_interpolates() -> None:
    import math

    import pytest

    from src.tournament.objective.cutoff_auc import CUTOFF_AUC_HIGH, CUTOFF_AUC_LOW, smooth_cutoff_utility

    assert CUTOFF_AUC_LOW == 0.40
    assert CUTOFF_AUC_HIGH == 0.55
    assert smooth_cutoff_utility(0.39) == 0.0
    assert smooth_cutoff_utility(0.40) == 0.0
    assert smooth_cutoff_utility(0.55) == 1.0
    assert smooth_cutoff_utility(0.80) == 1.0
    assert math.isclose(smooth_cutoff_utility(0.4782), (0.4782 - 0.40) / 0.15, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(smooth_cutoff_utility(0.5001), (0.5001 - 0.40) / 0.15, rel_tol=0.0, abs_tol=1e-12)
    with pytest.raises(ValueError):
        smooth_cutoff_utility(float("nan"))
    with pytest.raises(ValueError):
        smooth_cutoff_utility(0.5, low=0.55, high=0.40)


def test_cutoff_auc_score_equal_weight_grid() -> None:
    import pytest

    from src.tournament.objective.cutoff_auc import CUTOFF_AUC_THRESHOLDS, cutoff_auc_score

    assert CUTOFF_AUC_THRESHOLDS == (0.40, 0.45, 0.50, 0.55, 0.60)
    returns = (0.61, 0.51, 0.41, 0.0)
    score = cutoff_auc_score(returns)
    expected = (3 / 4 + 2 / 4 + 2 / 4 + 1 / 4 + 1 / 4) / 5
    assert score == pytest.approx(expected)
    with pytest.raises(ValueError):
        cutoff_auc_score(())
    with pytest.raises(ValueError):
        cutoff_auc_score((0.1, float("inf")))
    with pytest.raises(ValueError):
        cutoff_auc_score(("bad", 0.1))
    with pytest.raises(ValueError):
        cutoff_auc_score((0.1,), thresholds=(float("inf"),))
    with pytest.raises(ValueError):
        cutoff_auc_score((0.1,), thresholds=("bad",))


def test_mean_smooth_cutoff_utility_prefers_near_winner_cutoff() -> None:
    import math

    import pytest

    from src.tournament.objective.cutoff_auc import mean_smooth_cutoff_utility, smooth_cutoff_utility

    with pytest.raises(ValueError):
        mean_smooth_cutoff_utility(())
    with pytest.raises(ValueError):
        mean_smooth_cutoff_utility(("bad",))
    with pytest.raises(ValueError):
        mean_smooth_cutoff_utility((float("inf"),))

    winner = mean_smooth_cutoff_utility((0.4782,))
    just_over = mean_smooth_cutoff_utility((0.5001,))
    assert math.isclose(winner, smooth_cutoff_utility(0.4782), rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(just_over, smooth_cutoff_utility(0.5001), rel_tol=0.0, abs_tol=1e-12)
    assert 0.0 < just_over - winner < 0.20


def test_opportunity_conditioned_capture_length_mismatch_and_zero_oracle() -> None:
    import pytest

    from src.tournament.objective.cutoff_auc import opportunity_conditioned_capture

    assert opportunity_conditioned_capture((0.60, 0.10), (0.55, 0.51), threshold=0.50) == pytest.approx(0.5)
    assert opportunity_conditioned_capture((0.60, 0.10), (0.20, 0.10), threshold=0.50) == 0.0
    with pytest.raises(ValueError):
        opportunity_conditioned_capture((0.60,), (0.55, 0.51), threshold=0.50)


def test_inactive_false_positive_loss_cash_is_zero_and_drawdown_counts() -> None:
    import pytest

    from src.tournament.objective.cutoff_auc import inactive_activity_rate, inactive_false_positive_loss

    returns = (0.60, 0.0, -0.20, 0.0)
    active = (True, False, False, False)
    assert inactive_false_positive_loss(returns, active) == pytest.approx(0.20 / 3.0)
    assert inactive_activity_rate(returns, active) == pytest.approx(1 / 3)
    cash = (0.60, 0.0, 0.0)
    cash_active = (True, False, False)
    assert inactive_false_positive_loss(cash, cash_active) == 0.0
    assert inactive_activity_rate(cash, cash_active) == 0.0
    with pytest.raises(ValueError):
        inactive_false_positive_loss((0.1,), (True, False))
    assert inactive_false_positive_loss((0.60,), (True,)) == 0.0
    assert inactive_activity_rate((0.60,), (True,)) == 0.0
    with pytest.raises(ValueError):
        inactive_activity_rate((0.1,), (True, False))


def test_resolve_attack_sleeve_fail_closed_unknown() -> None:
    from src.tournament.championship_regime import ChampionshipSleeve
    from src.tournament.objective.cutoff_auc import resolve_attack_sleeve

    assert resolve_attack_sleeve(ChampionshipSleeve.LOTTERY_ON.value) == "MOM60"
    assert resolve_attack_sleeve(ChampionshipSleeve.CRASH_REBOUND.value) == "REBOUND"
    assert resolve_attack_sleeve(ChampionshipSleeve.INACTIVE.value) == "CASH"
    assert resolve_attack_sleeve(ChampionshipSleeve.UNCERTAIN.value) == "CASH"
    assert resolve_attack_sleeve(None) == "CASH"
    assert resolve_attack_sleeve("") == "CASH"
    assert resolve_attack_sleeve("not_a_sleeve") == "CASH"


def test_apply_attack_sleeve_route_identity_when_gate_false() -> None:
    from src.tournament.championship_regime import ChampionshipSleeve
    from src.tournament.objective.cutoff_auc import CUTOFF_AUC_IS_PRODUCTION_GATE, apply_attack_sleeve_route

    mom60 = {"AAA": 0.12}
    rebound = {"BBB": 0.20}
    assert CUTOFF_AUC_IS_PRODUCTION_GATE is False
    for sleeve in (ChampionshipSleeve.LOTTERY_ON.value, ChampionshipSleeve.CRASH_REBOUND.value, ChampionshipSleeve.INACTIVE.value, None):
        out = apply_attack_sleeve_route(sleeve=sleeve, mom60_scores=mom60, rebound_scores=rebound, production_gate=False)
        assert out == mom60


def test_apply_attack_sleeve_route_cash_rebound_mom60_when_gate_true() -> None:
    from src.portfolio.intent import CASH_INTENT
    from src.tournament.championship_regime import ChampionshipSleeve
    from src.tournament.objective.cutoff_auc import apply_attack_sleeve_route

    mom60 = {"AAA": 0.12}
    rebound = {"BBB": 0.20}
    lottery = apply_attack_sleeve_route(sleeve=ChampionshipSleeve.LOTTERY_ON.value, mom60_scores=mom60, rebound_scores=rebound, production_gate=True)
    crash = apply_attack_sleeve_route(sleeve=ChampionshipSleeve.CRASH_REBOUND.value, mom60_scores=mom60, rebound_scores=rebound, production_gate=True)
    empty_crash = apply_attack_sleeve_route(sleeve=ChampionshipSleeve.CRASH_REBOUND.value, mom60_scores=mom60, rebound_scores={}, production_gate=True)
    inactive = apply_attack_sleeve_route(sleeve=ChampionshipSleeve.INACTIVE.value, mom60_scores=mom60, rebound_scores=rebound, production_gate=True)
    assert lottery == mom60
    assert crash == rebound
    assert empty_crash is CASH_INTENT
    assert inactive is CASH_INTENT


def test_evaluate_attack_policy_hard_constraints_and_active_utility() -> None:
    from src.tournament.objective.cutoff_auc import evaluate_attack_policy
    from src.tournament.objective.reports import GROSS_METRIC_UNAVAILABLE

    candidate = (0.60, 0.0, 0.0, 0.50)
    incumbent = (0.10, 0.0, 0.0, 0.10)
    oracle = (0.70, 0.05, 0.02, 0.55)
    active = (True, False, False, True)
    missing = evaluate_attack_policy(candidate_returns=candidate, incumbent_returns=incumbent, oracle_returns=oracle, active=active, execution_parity=True, gross_violation_count=None)
    assert missing.status == "INSUFFICIENT_EVIDENCE"
    assert GROSS_METRIC_UNAVAILABLE in missing.failures
    hard = evaluate_attack_policy(candidate_returns=candidate, incumbent_returns=incumbent, oracle_returns=oracle, active=active, execution_parity=False, gross_violation_count=2)
    assert hard.status == "FAIL"
    assert "EXECUTION_PARITY" in hard.failures
    assert "GROSS_EXPOSURE" in hard.failures
    ruin = evaluate_attack_policy(candidate_returns=(0.60, -0.40, -0.40, 0.50), incumbent_returns=incumbent, oracle_returns=oracle, active=active, execution_parity=True, gross_violation_count=0)
    assert ruin.status == "FAIL"
    assert "RUIN" in ruin.failures
    worse = evaluate_attack_policy(candidate_returns=(0.10, 0.0, 0.0, 0.10), incumbent_returns=(0.60, 0.0, 0.0, 0.50), oracle_returns=oracle, active=active, execution_parity=True, gross_violation_count=0)
    assert worse.status == "FAIL"
    assert "ACTIVE_UTILITY" in worse.failures
    ok = evaluate_attack_policy(candidate_returns=candidate, incumbent_returns=incumbent, oracle_returns=oracle, active=active, execution_parity=True, gross_violation_count=0)
    assert ok.status == "PASS"
    assert ok.failures == ()


def test_evaluate_attack_policy_does_not_require_all_era_p50() -> None:
    from src.tournament.objective.cutoff_auc import evaluate_attack_policy

    candidate = (0.48, 0.0, 0.0, 0.47)
    incumbent = (0.10, 0.0, 0.0, 0.10)
    oracle = (0.60, 0.10, 0.10, 0.55)
    active = (True, False, False, True)
    assert sum(1 for value in candidate if value > 0.50) == 0
    result = evaluate_attack_policy(candidate_returns=candidate, incumbent_returns=incumbent, oracle_returns=oracle, active=active, execution_parity=True, gross_violation_count=0)
    assert result.status == "PASS"
    assert "SCENARIO_CHAMPIONSHIP_VS_INCUMBENT" not in result.failures
    assert "PRIMARY_CI_VS_INCUMBENT" not in result.failures


def test_evaluate_attack_policy_inactive_activity_fails() -> None:
    from src.tournament.objective.cutoff_auc import evaluate_attack_policy

    result = evaluate_attack_policy(
        candidate_returns=(0.60, 0.12, 0.0, 0.50),
        incumbent_returns=(0.10, 0.0, 0.0, 0.10),
        oracle_returns=(0.70, 0.05, 0.02, 0.55),
        active=(True, False, False, True),
        execution_parity=True,
        gross_violation_count=0,
    )
    assert result.status == "FAIL"
    assert "INACTIVE_ACTIVITY" in result.failures


def test_evaluate_attack_policy_empty_mismatch_insufficient() -> None:
    from src.tournament.objective.cutoff_auc import evaluate_attack_policy

    empty = evaluate_attack_policy(candidate_returns=(), incumbent_returns=(), oracle_returns=(), active=(), execution_parity=True, gross_violation_count=0)
    assert empty.status == "INSUFFICIENT_EVIDENCE"
    assert "MISSING_ARTIFACT" in empty.failures
    mismatch = evaluate_attack_policy(candidate_returns=(0.1,), incumbent_returns=(0.1, 0.2), oracle_returns=(0.1,), active=(True,), execution_parity=True, gross_violation_count=0)
    assert mismatch.status == "INSUFFICIENT_EVIDENCE"
    assert "MISSING_ARTIFACT" in mismatch.failures
    no_active = evaluate_attack_policy(candidate_returns=(0.0, 0.0), incumbent_returns=(0.0, 0.0), oracle_returns=(0.1, 0.1), active=(False, False), execution_parity=True, gross_violation_count=0)
    assert no_active.status == "INSUFFICIENT_EVIDENCE"
    assert "INSUFFICIENT_ACTIVE" in no_active.failures
    bad_type = evaluate_attack_policy(candidate_returns=("bad",), incumbent_returns=(0.1,), oracle_returns=(0.1,), active=(True,), execution_parity=True, gross_violation_count=0)
    assert bad_type.status == "INSUFFICIENT_EVIDENCE"
    assert "MISSING_ARTIFACT" in bad_type.failures
    bad_finite = evaluate_attack_policy(candidate_returns=(float("inf"),), incumbent_returns=(0.1,), oracle_returns=(0.1,), active=(True,), execution_parity=True, gross_violation_count=0)
    assert bad_finite.status == "INSUFFICIENT_EVIDENCE"
    assert "MISSING_ARTIFACT" in bad_finite.failures
