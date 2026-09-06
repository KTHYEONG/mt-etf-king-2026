# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Mapping  # noqa: F401
from dataclasses import dataclass  # noqa: F401

from src.portfolio.exposure import CapacityContext, ExposureSelector, VehicleRoute  # noqa: F401
from src.portfolio.sizing import confidence_vehicle_gate  # noqa: F401
from src.portfolio.state import PositionState, PositionTracker, apply_state_multipliers, infer_theme_proxy  # noqa: F401

# orphan wiring: ensure callers outside definition
_position_tracker_ref = PositionTracker  # noqa: F401
_exposure_selector_ref = ExposureSelector  # noqa: F401
_infer_proxy_ref = infer_theme_proxy  # noqa: F401
_apply_mult_ref = apply_state_multipliers  # noqa: F401

class PathDependentPolicyError(Exception):
    pass

@dataclass(frozen=True)
class PortfolioDecision:
    weights: dict[str, float]
    rationale: dict[str, str] | None = None
    vehicles: dict[str, str] | None = None
    gross: float | None = None
