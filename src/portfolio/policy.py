# ruff: noqa
# mypy: ignore-errors
"""Backward-compatible facade over split portfolio policy submodules."""

from __future__ import annotations

from src.portfolio.policy_core import PortfolioPolicy
from src.portfolio.policy_tail import PortfolioPolicyTailMixin
from src.portfolio.policy_types import (
    CapacityContext,
    ExposureSelector,
    PathDependentPolicyError,
    PortfolioDecision,
    PositionState,
    PositionTracker,
    VehicleRoute,
    _apply_mult_ref,
    _exposure_selector_ref,
    _infer_proxy_ref,
    _position_tracker_ref,
    apply_state_multipliers,
    confidence_vehicle_gate,
    infer_theme_proxy,
)

__all__ = [
    "CapacityContext",
    "ExposureSelector",
    "PathDependentPolicyError",
    "PortfolioDecision",
    "PortfolioPolicy",
    "PortfolioPolicyTailMixin",
    "PositionState",
    "PositionTracker",
    "VehicleRoute",
    "_apply_mult_ref",
    "_exposure_selector_ref",
    "_infer_proxy_ref",
    "_position_tracker_ref",
    "apply_state_multipliers",
    "confidence_vehicle_gate",
    "infer_theme_proxy",
]
