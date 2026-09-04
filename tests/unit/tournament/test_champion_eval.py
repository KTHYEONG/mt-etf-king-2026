from __future__ import annotations


def test_champion_promotion_requires_dual_scenario_loyo_and_integrity() -> None:
    from src.tournament.champion_eval import is_promotable

    assert is_promotable(aggressive_status='PASS', conservative_status='PASS', loyo_status='PASS', artifact_integrity=True) is True
    assert is_promotable(aggressive_status='PASS', conservative_status='FAIL', loyo_status='PASS', artifact_integrity=True) is False
    assert is_promotable(aggressive_status='PASS', conservative_status='PASS', loyo_status='FAIL', artifact_integrity=True) is False
    assert is_promotable(aggressive_status='PASS', conservative_status='PASS', loyo_status='PASS', artifact_integrity=False) is False
