"""Config-routing regression for the hardcoded-threshold architecture-lint fix.

Proves that moving CUTOFF_AUC_THRESHOLDS/LOW/HIGH, championship_regime's breadth-majority
threshold, and the P36 tournament horizon from Python literals to configs/gates.yaml /
configs/tournament.yaml changed zero numeric behavior.
"""


def test_cutoff_auc_and_regime_thresholds_config_sourced_values_unchanged() -> None:
    from src.tournament.championship_regime import _BREADTH_MAJORITY_THRESHOLD
    from src.tournament.objective.cutoff_auc import CUTOFF_AUC_HIGH, CUTOFF_AUC_LOW, CUTOFF_AUC_THRESHOLDS
    from src.tournament.objective_core import TOURNAMENT_SESSIONS

    assert CUTOFF_AUC_THRESHOLDS == (0.40, 0.45, 0.50, 0.55, 0.60)
    assert CUTOFF_AUC_LOW == 0.40
    assert CUTOFF_AUC_HIGH == 0.55
    assert _BREADTH_MAJORITY_THRESHOLD == 0.50
    assert TOURNAMENT_SESSIONS == 36


def test_cutoff_auc_thresholds_match_pre_config_routing_values() -> None:
    from src.tournament.objective.cutoff_auc import CUTOFF_AUC_HIGH, CUTOFF_AUC_LOW, CUTOFF_AUC_THRESHOLDS

    assert CUTOFF_AUC_THRESHOLDS == (0.40, 0.45, 0.50, 0.55, 0.60)
    assert CUTOFF_AUC_LOW == 0.40
    assert CUTOFF_AUC_HIGH == 0.55


def test_championship_regime_breadth_majority_matches_pre_config_routing_value() -> None:
    from src.tournament.championship_regime import _BREADTH_MAJORITY_THRESHOLD

    assert _BREADTH_MAJORITY_THRESHOLD == 0.50


def test_frontier_and_adaptive_specialists_horizon_matches_tournament_sessions() -> None:
    from src.tournament.objective_core import TOURNAMENT_SESSIONS

    assert TOURNAMENT_SESSIONS == 36
