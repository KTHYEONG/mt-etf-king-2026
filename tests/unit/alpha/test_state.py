"""SCENARIO-07-04 SCENARIO-07-05"""
from __future__ import annotations

import pytest

from src.alpha.state import ThemeMetrics, ThemeState, TransitionConfig, transition
from tests.unit.alpha.conftest import transition_config


def test_SCENARIO_07_04_transition_config_hysteresis() -> None:  # noqa: N802
    """SCENARIO-07-04"""
    cfg = TransitionConfig(
        rs_in=0.35,
        rs_out=0.55,
        rs_hi=0.75,
        accel_in=-0.08,
        accel_out=-0.20,
        breadth_in=0.65,
        breadth_out=0.45,
        ext_in=2.5,
        ext_out=1.5,
        dd_in=0.08,
        dd_out=0.05,
        patience=3,
    )
    with pytest.raises(ValueError, match="hysteresis"):
        cfg.validate()


def test_SCENARIO_07_05_breakdown_to_recovery_not_leading() -> None:  # noqa: N802
    """SCENARIO-07-05"""
    cfg = transition_config()
    metrics = ThemeMetrics(
        theme="T1",
        representative="A",
        rs=0.80,
        accel=0.05,
        breadth=0.70,
        ext=1.0,
        dd=0.01,
    )
    next_state, _ = transition(ThemeState.BREAKDOWN, metrics, cfg)
    assert next_state in (ThemeState.RECOVERY, ThemeState.BREAKDOWN)
    assert next_state != ThemeState.LEADING
