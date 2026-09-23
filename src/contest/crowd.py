"""Calibrated contest crowd: exposure tiers, behaviour mix, and leaderboard pinning.

Only ~1% of participants trade concentrated at ~full weight, ~5-15% partial,
the rest diversified; this crowd reproduces the 2025 contest (winner 47.8%,
2nd 44.6%) and the 2026 day-3 top-50. Only the 자율형 division may use
leverage/inverse; other divisions hold longs at half weight (= 1x on a
daily-reset 2x vehicle).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from src.contest.engine import CrowdSpec, SimState, Worlds
from src.contest.panel import VehiclePanel

logger = logging.getLogger(__name__)

_MOMENTUM_KS: Final[tuple[int, ...]] = (3, 5, 10, 20)
_MOMENTUM_QS: Final[tuple[float, ...]] = (0.05, 0.1, 0.2)
_CONTRARIAN_KS: Final[tuple[int, ...]] = (5, 10)
_CONTRARIAN_QS: Final[tuple[float, ...]] = (0.05, 0.1)
_RANDOM_QS: Final[tuple[float, ...]] = (0.03, 0.08)


def _normalized(mapping: Mapping[str, float], label: str) -> dict[str, float]:
    total = sum(float(v) for v in mapping.values())
    if total <= 0:
        raise ValueError(f"{label} weights sum to non-positive")
    return {k: float(v) / total for k, v in mapping.items()}


def build_crowd(
    panel: VehiclePanel,
    worlds: Worlds,
    profile: Mapping[str, Any],
    popularity_auto: Mapping[str, float],
    popularity_nonauto: Mapping[str, float],
    n_participants: int,
    f_auto: float,
    entry_days: Sequence[int],
    entry_probs: Sequence[float],
    seed: int,
) -> CrowdSpec:
    """Calibrated crowd: per division (자율형 leverage/inverse allowed; other divisions longs at weight x0.5 = 1x),
    aggressive tier weight U(0.85,1), moderate U(0.3,0.7), diversified U(0,0.3); behaviour mix per profile."""
    rng = np.random.default_rng(seed)
    names = set(panel.names)
    pop_auto = {a: w for a, w in _normalized(popularity_auto, "popularity_auto").items() if a in names}
    pop_non = {a: w for a, w in _normalized(popularity_nonauto, "popularity_nonauto").items() if a in names}
    pop_non = {a: w for a, w in pop_non.items() if not a.endswith("I")}
    if not pop_auto or not pop_non:
        raise ValueError("popularity maps empty after panel/inverse filtering")
    set_auto = np.array([panel.index(a) for a in pop_auto], dtype=np.int16)
    set_non = np.array([panel.index(a) for a in pop_non], dtype=np.int16)
    sets = [set_auto, set_non, set_auto]
    f_aggr = float(profile["f_aggr"])
    f_mod = float(profile["f_mod"])
    mix_aggr = [float(x) for x in profile["mix_aggr"]]
    mix_mod = [float(x) for x in profile["mix_mod"]]
    q_scale = float(profile["q_scale"])
    entry_p = np.asarray([float(x) for x in entry_probs], dtype=float)
    entry_p = entry_p / entry_p.sum()
    entry_d = np.asarray(list(entry_days))
    if n_participants <= 0:
        raise ValueError(f"non-positive n_participants: {n_participants}")

    kind: list[int] = []
    entry: list[int] = []
    weight: list[float] = []
    kk: list[int] = []
    qq: list[float] = []
    sid: list[int] = []
    is_auto: list[bool] = []

    def add(n: int, mix: Sequence[float], w_lo: float, w_hi: float, auto: bool) -> None:
        counts = np.floor(np.array([float(x) for x in mix]) * n).astype(int)
        counts[0] += n - counts.sum()
        for kind_id, cnt in enumerate(counts):
            kind.extend([kind_id] * cnt)
            is_auto.extend([auto] * cnt)
            entry.extend(rng.choice(entry_d, size=cnt, p=entry_p).tolist())
            scale = 1.0 if auto else 0.5
            weight.extend((rng.uniform(w_lo, w_hi, cnt) * scale).tolist())
            if kind_id == 1:
                kk.extend(rng.choice(_MOMENTUM_KS, cnt).tolist())
                qq.extend((q_scale * rng.choice(_MOMENTUM_QS, cnt)).tolist())
            elif kind_id == 2:
                kk.extend(rng.choice(_CONTRARIAN_KS, cnt).tolist())
                qq.extend((q_scale * rng.choice(_CONTRARIAN_QS, cnt)).tolist())
            elif kind_id == 3:
                kk.extend([0] * cnt)
                qq.extend((q_scale * rng.choice(_RANDOM_QS, cnt)).tolist())
            else:
                kk.extend([0] * cnt)
                qq.extend([0.0] * cnt)
            sid.extend([(0 if kind_id != 3 else 2) if auto else 1] * cnt)

    n_auto = int(round(n_participants * f_auto))
    for n_div, auto in ((n_auto, True), (n_participants - n_auto, False)):
        n_a = int(round(n_div * f_aggr))
        n_m = int(round(n_div * f_mod))
        add(n_a, mix_aggr, 0.85, 1.0, auto)
        add(n_m, mix_mod, 0.3, 0.7, auto)
        add(n_div - n_a - n_m, (1.0, 0.0, 0.0, 0.0), 0.0, 0.3, auto)

    kind_a = np.array(kind, dtype=np.int8)
    auto_a = np.array(is_auto)
    hold_idx = np.where(kind_a == 0)[0]
    n_worlds = worlds.n_worlds
    per_world = np.empty((len(hold_idx), n_worlds), dtype=np.int16)
    for auto in (True, False):
        rows_h = np.where(auto_a[hold_idx] == auto)[0]
        if not len(rows_h):
            continue
        pop = pop_auto if auto else pop_non
        vehicles = np.array([panel.index(a) for a in pop], dtype=np.int16)
        probs = np.array([pop[a] for a in pop], dtype=float)
        probs = probs / probs.sum()
        per_world[rows_h] = vehicles[rng.choice(len(vehicles), size=(len(rows_h), n_worlds), p=probs)]
    spec = CrowdSpec(
        kind=kind_a,
        vehicle=np.zeros(len(kind_a), dtype=np.int16),
        entry=np.array(entry),
        weight=np.array(weight, dtype=np.float32),
        k=np.array(kk),
        q=np.array(qq, dtype=np.float32),
        set_id=np.array(sid),
        sets=sets,
    )
    spec.vehicle_per_world = per_world
    logger.info(f"[DATA] contest_crowd agents={len(kind_a)} auto={n_auto} hold={len(hold_idx)}")
    return spec


def pin_to_leaderboard(state: "SimState", top_values: np.ndarray, explicit: Sequence[tuple[int, float, int]]) -> None:
    """At the last realized session: explicit inferred leaders (agent index, equity, vehicle) get their real equity and
    holding; the remaining real top-50 values are assigned to the model's best agents; every other agent is capped at
    the real 50th value (the observed field is thinner than the unconstrained model)."""
    top = np.asarray(top_values, dtype=np.float32).ravel()
    if top.size == 0:
        raise ValueError("empty top_values")
    ceiling = float(top[0])
    floor = float(top[-1])
    eq = state.eq
    np.minimum(eq, ceiling, out=eq)
    remaining = top.tolist()
    for _idx, val, _veh in explicit:
        for j, cand in enumerate(remaining):
            if abs(cand - val) <= 1e-6 * max(1.0, abs(val)):
                remaining.pop(j)
                break
    if explicit:
        eq[[idx for idx, _, _ in explicit]] = -np.inf
    order = np.argsort(-eq, axis=0)
    ar_w = np.arange(eq.shape[1])
    np.minimum(eq, floor, out=eq)
    assign = np.asarray(remaining, dtype=np.float32)
    if assign.size:
        take = min(assign.size, eq.shape[0] - len(explicit))
        eq[order[:take], ar_w[None, :]] = assign[:take, None]
    for idx, val, veh in explicit:
        eq[idx] = np.float32(val)
        state.held[idx] = np.int16(veh)
    np.minimum(eq, ceiling, out=eq)


def append_explicit_leaders(
    crowd: CrowdSpec, holders: Sequence[tuple[float, int]], churn_k: int, churn_q: float
) -> tuple[CrowdSpec, list[int]]:
    """Add one momentum-churn agent per inferred single-vehicle leader (entry day 0) and return their indices."""
    start = crowd.n_agents
    new_indices = list(range(start, start + len(holders)))
    if not holders:
        return crowd, []
    extended = CrowdSpec(
        kind=np.concatenate([crowd.kind, np.ones(len(holders), dtype=np.int8)]),
        vehicle=np.concatenate(
            [crowd.vehicle, np.array([v for _, v in holders], dtype=np.int16)]
        ),
        entry=np.concatenate([crowd.entry, np.zeros(len(holders), dtype=crowd.entry.dtype)]),
        weight=np.concatenate(
            [crowd.weight, np.array([w for w, _ in holders], dtype=np.float32)]
        ),
        k=np.concatenate([crowd.k, np.full(len(holders), churn_k, dtype=crowd.k.dtype)]),
        q=np.concatenate([crowd.q, np.full(len(holders), churn_q, dtype=np.float32)]),
        set_id=np.concatenate([crowd.set_id, np.zeros(len(holders), dtype=crowd.set_id.dtype)]),
        sets=crowd.sets,
        vehicle_per_world=crowd.vehicle_per_world,
    )
    return extended, new_indices
