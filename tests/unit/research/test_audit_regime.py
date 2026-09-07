from src.research.audit_regime import AUDIT_REGIME_IS_ACTIVATION_GATE, audit_regime_label


def test_audit_regime_label_frozen_thresholds_are_not_activation_gate() -> None:
    assert AUDIT_REGIME_IS_ACTIVATION_GATE is False
    assert audit_regime_label(mom60=-0.22, mom20=0.23, rv20=0.90, dd60=-0.26) == "high_vol_reversal"
    assert audit_regime_label(mom60=0.10, mom20=0.02, rv20=0.10, dd60=-0.01) == "strong_risk_on"
    assert audit_regime_label(mom60=0.02, mom20=0.01, rv20=0.10, dd60=-0.02) == "weak_risk_on"
    assert audit_regime_label(mom60=-0.10, mom20=-0.01, rv20=0.10, dd60=-0.04) == "risk_off"
    assert audit_regime_label(mom60=-0.02, mom20=0.0, rv20=0.10, dd60=-0.04) == "sideways"
    assert audit_regime_label(mom60=0.09, mom20=0.01, rv20=0.33, dd60=-0.01) == "high_vol_reversal"
