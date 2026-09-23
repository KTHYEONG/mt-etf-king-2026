"""Invariant guards for the contest crowd model."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.contest.crowd import append_explicit_leaders, build_crowd, pin_to_leaderboard
from src.contest.engine import CrowdSpec, SimState, bootstrap_worlds
from src.contest.panel import VehiclePanel

NAMES = ("K2", "Q2", "SEMI2", "SEMI2b", "HY2", "SS2", "BAT2", "SHIP2", "K2I", "HY2I", "Q2I", "BIO2", "FIN2", "IND2")
POP_AUTO = {"SEMI2": 0.22, "K2": 0.22, "HY2": 0.14, "HY2I": 0.05, "SS2": 0.07, "SEMI2b": 0.06, "K2I": 0.05, "Q2": 0.06,
            "BAT2": 0.04, "SHIP2": 0.04, "Q2I": 0.05}
POP_NON = {"SEMI2": 0.30, "K2": 0.25, "SS2": 0.10, "HY2": 0.08, "BAT2": 0.07, "SHIP2": 0.05, "Q2": 0.05,
           "BIO2": 0.03, "FIN2": 0.03, "IND2": 0.04}
PROFILE = {"f_aggr": 0.01, "f_mod": 0.15, "mix_aggr": [0.45, 0.30, 0.10, 0.15],
           "mix_mod": [0.60, 0.20, 0.10, 0.10], "q_scale": 1.0}
ENTRY_DAYS = [0, 1, 2, 3, 5, 8]
ENTRY_PROBS = [0.70, 0.13, 0.07, 0.04, 0.03, 0.03]


def _toy_panel() -> VehiclePanel:
    t = 100
    dates = tuple(date(2026, 1, 1) + timedelta(days=i) for i in range(t))
    rng = np.random.default_rng(0)
    gap = rng.uniform(-0.02, 0.02, (t, len(NAMES))).astype(np.float32)
    intra = rng.uniform(-0.02, 0.02, (t, len(NAMES))).astype(np.float32)
    gap[0] = 0.0
    cc = (1 + gap.astype(np.float64)) * (1 + intra.astype(np.float64)) - 1.0
    return VehiclePanel(dates=dates, names=NAMES, gap=gap, intraday=intra, log_nav=np.cumsum(np.log1p(cc), axis=0))


def _toy_worlds():
    return bootstrap_worlds(_toy_panel(), [70, 71], np.arange(0, 70), 3, 4, 10.0, 0)


def _crowd(n: int = 1000, f_auto: float = 0.4, seed: int = 11) -> CrowdSpec:
    return build_crowd(_toy_panel(), _toy_worlds(), PROFILE, POP_AUTO, POP_NON, n, f_auto, ENTRY_DAYS, ENTRY_PROBS, seed)


def _is_auto(crowd: CrowdSpec) -> np.ndarray:
    return crowd.weight > 0.5


def test_build_division_constraints() -> None:
    """Non-자율 agents hold at most half weight and never touch inverse vehicles."""
    crowd = build_crowd(_toy_panel(), _toy_worlds(), PROFILE, POP_AUTO, POP_NON, 1000, 0.0, ENTRY_DAYS, ENTRY_PROBS, 5)
    assert crowd.n_agents == 1000
    assert crowd.weight.max() <= 0.5 + 1e-9
    inverse_ids = {NAMES.index(n) for n in NAMES if n.endswith("I")}
    assert not (set(crowd.sets[1].tolist()) & inverse_ids)
    per_world = crowd.vehicle_per_world
    assert per_world is not None
    assert not (set(np.unique(per_world).tolist()) & inverse_ids)


def test_build_pin_assigns_real_top_values() -> None:
    """Explicit leaders keep real equity; best model agents take the rest in order; others capped."""
    rng = np.random.default_rng(2)
    n_agents, n_worlds = 20, 2
    eq = rng.uniform(0.9, 1.6, (n_agents, n_worlds)).astype(np.float32)
    held = np.zeros((n_agents, n_worlds), dtype=np.int16)
    state = SimState(day=2, eq=eq, held=held)
    top = np.array([1.60, 1.50, 1.40, 1.30, 1.20], dtype=np.float32)
    pin_to_leaderboard(state, top, [(0, 1.60, 3)])
    assert (state.eq[0] == np.float32(1.60)).all()
    assert (state.held[0] == 3).all()
    assert state.eq.max() <= np.float32(1.60)
    for w in range(n_worlds):
        best5 = np.sort(state.eq[:, w])[-5:]
        assert np.allclose(best5, [1.20, 1.30, 1.40, 1.50, 1.60], atol=1e-6)


def test_build_tier_counts_follow_profile() -> None:
    """Aggressive 자율 agents number round(n*f_auto*f_aggr) with weights in [0.85, 1]."""
    crowd = _crowd(n=1000, f_auto=0.4)
    n_auto = int(round(1000 * 0.4))
    n_aggr = int(round(n_auto * 0.01))
    assert n_aggr == 4
    auto_aggr = [i for i in range(n_auto) if crowd.weight[i] >= 0.85]
    assert len(auto_aggr) == n_aggr
    assert ((crowd.weight[auto_aggr] >= 0.85) & (crowd.weight[auto_aggr] <= 1.0)).all()


def test_append_explicit_leaders_adds_churn_agents() -> None:
    """One entry-day-0 momentum agent per inferred leader, with churn k/q."""
    crowd = _crowd(n=200)
    before = crowd.n_agents
    v0, v1 = NAMES.index("SEMI2"), NAMES.index("HY2")
    extended, indices = append_explicit_leaders(crowd, [(1.0, v0), (0.5, v1)], 10, 0.10)
    assert indices == [before, before + 1]
    assert extended.n_agents == before + 2
    assert (extended.kind[indices] == 1).all()
    assert (extended.k[indices] == 10).all()
    assert np.allclose(extended.q[indices], 0.10)
    assert (extended.entry[indices] == 0).all()
    assert extended.vehicle[before] == v0 and extended.vehicle[before + 1] == v1
    empty, none = append_explicit_leaders(crowd, [], 10, 0.10)
    assert none == [] and empty.n_agents == before


def test_build_rejects_degenerate_inputs() -> None:
    """Zero-sum popularity, fully filtered maps, and non-positive sizes fail closed."""
    panel, worlds = _toy_panel(), _toy_worlds()
    with pytest.raises(ValueError, match=r"non-positive"):
        build_crowd(panel, worlds, PROFILE, {"K2": 0.0}, POP_NON, 100, 0.4, ENTRY_DAYS, ENTRY_PROBS, 0)
    with pytest.raises(ValueError, match=r"after panel"):
        build_crowd(panel, worlds, PROFILE, {"GHOST": 1.0}, POP_NON, 100, 0.4, ENTRY_DAYS, ENTRY_PROBS, 0)
    with pytest.raises(ValueError, match=r"non-positive n_participants"):
        build_crowd(panel, worlds, PROFILE, POP_AUTO, POP_NON, 0, 0.4, ENTRY_DAYS, ENTRY_PROBS, 0)
    with pytest.raises(ValueError, match=r"empty top_values"):
        pin_to_leaderboard(SimState(day=0, eq=np.ones((2, 1), dtype=np.float32),
                                    held=np.zeros((2, 1), dtype=np.int16)),
                           np.array([], dtype=np.float32), [])
