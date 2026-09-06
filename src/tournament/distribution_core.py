# mypy: ignore-errors
# ruff: noqa
"""Backward-compatible facade over src.tournament.distribution (P5 split).

Canonical home of every symbol below is the src.tournament.distribution
package (moved from distribution_core.py); this module re-exports them so existing import
paths keep working.
"""

from __future__ import annotations

from src.tournament.distribution import ReturnDistribution
from src.tournament.distribution import b1_gate_anchors_from_distribution
from src.tournament.distribution import championship_lock_returns
from src.tournament.distribution import continuation_capture
from src.tournament.distribution import effective_sample_size
from src.tournament.distribution import evaluate_adoption_gates
from src.tournament.distribution import evaluate_p24_adoption_gates
from src.tournament.distribution import evaluate_p25_adoption_gates
from src.tournament.distribution import evaluate_tail_gates
from src.tournament.distribution import exceedance_curve
from src.tournament.distribution import execution_faithful_late_lock_returns
from src.tournament.distribution import house_money_ratchet_returns
from src.tournament.distribution import locked_window_returns
from src.tournament.distribution import measure_vehicle_activity_from_allocate
from src.tournament.distribution import measure_vehicle_activity_from_session_cache
from src.tournament.distribution import measure_vehicle_activity_from_top1_scores
from src.tournament.distribution import oneshot_anchor_starts
from src.tournament.distribution import oneshot_window_returns
from src.tournament.distribution import overlay_right_tail_stats
from src.tournament.distribution import preflight_features_span_ok
from src.tournament.distribution import resolve_adoption_vehicle_rate
from src.tournament.distribution import right_tail_score
from src.tournament.distribution import ruin_probability
from src.tournament.distribution import score_seed_for_vehicle_probe
from src.tournament.distribution import serialize_oneshot_rows
from src.tournament.distribution import stationary_bootstrap_ci
from src.tournament.distribution import vehicle_activity_rate

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
