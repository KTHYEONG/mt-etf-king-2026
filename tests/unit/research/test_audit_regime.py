from src.research.audit_regime import AUDIT_REGIME_IS_ACTIVATION_GATE, audit_regime_label


def test_audit_regime_label_frozen_thresholds_are_not_activation_gate() -> None:
    assert AUDIT_REGIME_IS_ACTIVATION_GATE is False
    assert audit_regime_label(mom60=-0.22, mom20=0.23, rv20=0.90, dd60=-0.26) == "high_vol_reversal"
    assert audit_regime_label(mom60=0.10, mom20=0.02, rv20=0.10, dd60=-0.01) == "strong_risk_on"
    assert audit_regime_label(mom60=0.02, mom20=0.01, rv20=0.10, dd60=-0.02) == "weak_risk_on"
    assert audit_regime_label(mom60=-0.10, mom20=-0.01, rv20=0.10, dd60=-0.04) == "risk_off"
    assert audit_regime_label(mom60=-0.02, mom20=0.0, rv20=0.10, dd60=-0.04) == "sideways"
    assert audit_regime_label(mom60=0.09, mom20=0.01, rv20=0.33, dd60=-0.01) == "high_vol_reversal"


def test_audit_regime_annualize_then_label_hvr() -> None:
    from src.research.audit_regime import AUDIT_REGIME_IS_ACTIVATION_GATE, annualize_daily_realized_vol, audit_regime_label

    assert AUDIT_REGIME_IS_ACTIVATION_GATE is False
    assert audit_regime_label(mom60=-0.22, mom20=0.23, rv20=0.05644660723119608, dd60=-0.26) != "high_vol_reversal"
    annual = annualize_daily_realized_vol(0.05644660723119608)
    assert audit_regime_label(mom60=-0.22, mom20=0.23, rv20=annual, dd60=-0.26) == "high_vol_reversal"
