from __future__ import annotations


def test_champion_policy_demotes_unfillable_two_x_and_enforces_gross() -> None:
    from datetime import date
    from src.strategies.champion_tail import ChampionPolicyConfig, ChampionTailPolicy
    from src.universe.instruments import Confidence, InstrumentAttributes, InstrumentMaster

    attrs = {
        'ONE': InstrumentAttributes('ONE', 'KODEX 200', 'S', 1, 'KOSPI 200', False, False, True, 'KOSPI 200', 'EQUITY', date(2020, 1, 1), date(2026, 12, 31), False, Confidence.HIGH),
        'TWO': InstrumentAttributes('TWO', 'KODEX 레버리지', 'S', 2, 'KOSPI 200', False, False, True, 'KOSPI 200', 'EQUITY', date(2020, 1, 1), date(2026, 12, 31), False, Confidence.HIGH),
    }
    policy = ChampionTailPolicy(InstrumentMaster(attrs, date(2020, 1, 1)), ChampionPolicyConfig(max_single_weight=0.80, max_effective_gross=1.60, min_cash=0.05, absolute_momentum_cash=True))
    decision = policy.allocate({'ONE': 0.9}, capital=1_000_000_000.0, adv={'ONE': 1e12, 'TWO': 1e9}, participation=0.01, regime='RISK_ON', leverage_allowed=True, inverse_allowed=False, current_weights={})

    assert decision.weights == {'ONE': 0.80}
    assert decision.gross == 0.80
    assert sum(decision.weights.values()) <= 0.95
