from __future__ import annotations

from src.portfolio.sizing import (
    ConfidenceSizingConfig,
    TailConcentrationConfig,
    confidence_weights,
    tail_concentration_weights,
)


def test_tail_concentration_weights_leading_risk_on() -> None:
    base = ConfidenceSizingConfig()
    tail = TailConcentrationConfig(enabled=True)
    scores = {"TOP": 0.10, "OTHER": 0.04}
    result = tail_concentration_weights(
        scores,
        base,
        tail,
        theme_states={"TOP": "LEADING"},
        regime="RISK_ON",
    )
    assert result == {"TOP": 1.0}


def test_tail_concentration_weights_fallback_p11() -> None:
    base = ConfidenceSizingConfig()
    scores = {"TOP": 0.10, "OTHER": 0.04}
    enabled_tail = TailConcentrationConfig(enabled=True)
    disabled_tail = TailConcentrationConfig(enabled=False)
    theme_states = {"TOP": "EMERGING", "OTHER": "LEADING"}

    risk_off = tail_concentration_weights(scores, base, enabled_tail, theme_states, "RISK_OFF")
    emerging_only = tail_concentration_weights(
        scores,
        base,
        enabled_tail,
        theme_states,
        "RISK_ON",
    )
    disabled = tail_concentration_weights(scores, base, disabled_tail, theme_states, "RISK_ON")
    expected = confidence_weights(scores, base)

    for actual in (risk_off, emerging_only, disabled):
        assert set(actual.keys()) == set(expected.keys())
        for ticker, weight in expected.items():
            assert abs(float(actual[ticker]) - float(weight)) < 1e-9
