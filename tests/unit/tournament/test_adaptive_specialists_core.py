"""Direct coverage for src/tournament/adaptive_specialists_core.py, split out of adaptive_specialists.py
(400-statement budget, R6 AI-context invariant). Imports the core module directly (not the
re-export in adaptive_specialists.py) so a regression in the split itself, not just the
re-exported public surface, is caught here.
"""

from __future__ import annotations


def test_adaptive_specialist_config_horizon_matches_tournament_sessions_after_config_routing() -> None:
    # Given/When: the dataclass default is now sourced from TOURNAMENT_SESSIONS, not a literal 36
    from src.tournament.adaptive_specialists_core import AdaptiveSpecialistConfig
    from src.tournament.objective_core import TOURNAMENT_SESSIONS

    config = AdaptiveSpecialistConfig()

    # Then: the config-routed value is unchanged from the pre-routing literal
    assert config.horizon == 36
    assert config.horizon == TOURNAMENT_SESSIONS


def test_bucket_for_partitions_by_leverage_multiple_and_theme() -> None:
    from src.tournament.adaptive_specialists_core import _bucket_for

    assert _bucket_for(2, "KOSPI200", None) == "long_broad"
    assert _bucket_for(2, "SEMICONDUCTOR", None) == "long_theme"
    assert _bucket_for(-2, "KOSPI200", None) == "inverse"
    assert _bucket_for(-1, "SEMICONDUCTOR", None) == "inverse"
    assert _bucket_for(1, "BOND", None) == "defensive"
    assert _bucket_for(1, "SEMICONDUCTOR", None) is None
    assert _bucket_for(3, "KOSPI200", None) is None


def test_proposal_intent_cash_for_none_ticker_or_nonpositive_weight() -> None:
    from src.tournament.adaptive_specialists_core import SpecialistProposal, _proposal_intent
    from src.portfolio.intent import CASH_INTENT

    cash_ticker = SpecialistProposal("long_broad", None, 0.0, 0.0, (float("inf"), float("inf"), ""))
    zero_weight = SpecialistProposal("long_broad", "005930", 0.0, 0.0, (0.0, 0.0, "005930"))
    real = SpecialistProposal("long_broad", "005930", 0.80, 1.60, (0.0, 0.0, "005930"))

    assert _proposal_intent(cash_ticker) is CASH_INTENT
    assert _proposal_intent(zero_weight) is CASH_INTENT
    intent = _proposal_intent(real)
    assert intent.kind == "target"
    assert intent.weights == {"005930": 0.80}


def test_cash_proposals_covers_all_four_specialists_with_zero_weight() -> None:
    from src.tournament.adaptive_specialists_core import _ORDER, _cash_proposals

    proposals = _cash_proposals()

    assert tuple(p.specialist for p in proposals) == _ORDER
    assert all(p.ticker is None and p.target_weight == 0.0 and p.effective_gross == 0.0 for p in proposals)
