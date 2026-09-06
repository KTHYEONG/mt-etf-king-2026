# mypy: ignore-errors
# ruff: noqa
"""Return-distribution package (P5 split of distribution_core.py + overlay_returns.py)."""

from __future__ import annotations

from src.tournament.distribution.bootstrap import stationary_bootstrap_ci
from src.tournament.distribution.core import ReturnDistribution
from src.tournament.distribution.core import effective_sample_size
from src.tournament.distribution.core import exceedance_curve
from src.tournament.distribution.core import oneshot_anchor_starts
from src.tournament.distribution.core import oneshot_window_returns
from src.tournament.distribution.core import preflight_features_span_ok
from src.tournament.distribution.core import right_tail_score
from src.tournament.distribution.core import ruin_probability
from src.tournament.distribution.core import serialize_oneshot_rows
from src.tournament.distribution.exceedance import b1_gate_anchors_from_distribution
from src.tournament.distribution.exceedance import evaluate_adoption_gates
from src.tournament.distribution.exceedance import evaluate_tail_gates
from src.tournament.distribution.exceedance import measure_vehicle_activity_from_allocate
from src.tournament.distribution.exceedance import measure_vehicle_activity_from_session_cache
from src.tournament.distribution.exceedance import measure_vehicle_activity_from_top1_scores
from src.tournament.distribution.exceedance import resolve_adoption_vehicle_rate
from src.tournament.distribution.exceedance import score_seed_for_vehicle_probe
from src.tournament.distribution.exceedance import vehicle_activity_rate
from src.tournament.distribution.overlays import championship_lock_returns
from src.tournament.distribution.overlays import continuation_capture
from src.tournament.distribution.overlays import evaluate_p24_adoption_gates
from src.tournament.distribution.overlays import evaluate_p25_adoption_gates
from src.tournament.distribution.overlays import execution_faithful_late_lock_returns
from src.tournament.distribution.overlays import house_money_ratchet_returns
from src.tournament.distribution.overlays import locked_window_returns
from src.tournament.distribution.overlays import overlay_right_tail_stats

__all__ = [
    "ReturnDistribution",
    "b1_gate_anchors_from_distribution",
    "championship_lock_returns",
    "continuation_capture",
    "effective_sample_size",
    "evaluate_adoption_gates",
    "evaluate_p24_adoption_gates",
    "evaluate_p25_adoption_gates",
    "evaluate_tail_gates",
    "exceedance_curve",
    "execution_faithful_late_lock_returns",
    "house_money_ratchet_returns",
    "locked_window_returns",
    "measure_vehicle_activity_from_allocate",
    "measure_vehicle_activity_from_session_cache",
    "measure_vehicle_activity_from_top1_scores",
    "oneshot_anchor_starts",
    "oneshot_window_returns",
    "overlay_right_tail_stats",
    "preflight_features_span_ok",
    "resolve_adoption_vehicle_rate",
    "right_tail_score",
    "ruin_probability",
    "score_seed_for_vehicle_probe",
    "serialize_oneshot_rows",
    "stationary_bootstrap_ci",
    "vehicle_activity_rate",
]
