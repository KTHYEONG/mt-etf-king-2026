"""Frozen audit regime labels (research reproduction only, never an activation gate)."""

from __future__ import annotations

import math
from typing import Final

AUDIT_REGIME_IS_ACTIVATION_GATE: Final[bool] = False
AUDIT_RV20_SESSIONS_PER_YEAR: Final[float] = 252.0


def annualize_daily_realized_vol(daily_std: float | None) -> float | None:
    if daily_std is None:
        return None
    value = float(daily_std)
    if not math.isfinite(value) or value < 0:
        return None
    return float(value * (AUDIT_RV20_SESSIONS_PER_YEAR**0.5))


def _num(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def audit_regime_label(
    *, mom60: float | None, mom20: float | None, rv20: float | None, dd60: float | None
) -> str:
    m60 = _num(mom60)
    m20 = _num(mom20)
    rv = _num(rv20)
    dd = _num(dd60)
    if rv is not None and (
        (m20 is not None and m60 is not None and rv > 0.25 and m20 > 0.03 and m60 < 0) or rv > 0.32
    ):
        return "high_vol_reversal"
    if m60 is not None and m60 > 0.08 and (dd is None or dd > -0.05):
        return "strong_risk_on"
    if m60 is not None and m60 >= 0:
        return "weak_risk_on"
    if (m60 is not None and m60 < -0.08) or (dd is not None and dd < -0.15):
        return "risk_off"
    return "sideways"
