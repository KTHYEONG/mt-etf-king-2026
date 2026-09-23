"""Vectorized contest simulator: bootstrap worlds, crowd agents, and our actions.

A world is one market path of D sessions given by row indices into the vehicle
panel. Crowd agents are simulated jointly as [A, W] arrays; our candidate
actions run alongside and never affect the crowd. A switch executed at the open
earns the old holding's overnight gap and the new holding's intraday leg.
Arrays are indexed by panel rows (no [W, D, V] return cubes are materialized).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from src.contest.panel import VehiclePanel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Worlds:
    rows: np.ndarray  # [W, D] panel row index per simulated session
    log_nav0: np.ndarray  # [W, lookback, V] history before day 0 (momentum/volatility signals)

    @property
    def n_worlds(self) -> int:
        """Number of simulated worlds."""
        return int(self.rows.shape[0])

    @property
    def horizon(self) -> int:
        """Number of simulated sessions per world."""
        return int(self.rows.shape[1])


@dataclass
class CrowdSpec:
    """Per-agent parameters (kind 0 hold, 1 momentum, 2 contrarian, 3 random), weights, entry day, lookback k,
    daily switch probability q, candidate set id, and per-world hold vehicle for hold agents."""

    kind: np.ndarray  # [A] int8
    vehicle: np.ndarray  # [A] int16 hold vehicle for kind-0 agents
    entry: np.ndarray  # [A] int64 first active session
    weight: np.ndarray  # [A] float32 exposure weight
    k: np.ndarray  # [A] int64 momentum/contrarian lookback
    q: np.ndarray  # [A] float32 daily switch probability
    set_id: np.ndarray  # [A] int64 candidate set index
    sets: list[np.ndarray] = field(default_factory=list)
    vehicle_per_world: np.ndarray | None = None  # [A_hold, W] int16 per-world hold vehicle

    @property
    def n_agents(self) -> int:
        """Number of crowd agents."""
        return int(len(self.kind))


@dataclass
class SimState:
    """Mutable per-day crowd state handed to the end-of-day hook (pinned in place)."""

    day: int
    eq: np.ndarray  # [A, W] float32 crowd equity
    held: np.ndarray  # [A, W] int16 crowd holdings (-1 = cash)


@dataclass(frozen=True)
class SimResult:
    """Final equities: crowd [A, W] float32 and ours per action [W] float64."""

    crowd_equity: np.ndarray
    ours: dict[str, np.ndarray]


def bootstrap_worlds(
    panel: VehiclePanel,
    realized_rows: Sequence[int],
    pool_rows: np.ndarray,
    n_worlds: int,
    horizon: int,
    mean_block: float,
    seed: int,
) -> Worlds:
    """Realized contest sessions first (identical in every world), then a stationary block bootstrap of joint
    vehicle legs from `pool_rows` for the remaining `horizon` sessions."""
    from src.core.config import load_config

    lookback = int(load_config("contest").get("contest", {}).get("simulation", {}).get("lookback_sessions", 61))
    rng = np.random.default_rng(seed)
    pool = np.asarray(pool_rows, dtype=np.int64).ravel()
    if pool.size == 0:
        raise ValueError("empty pool_rows")
    if horizon < 0:
        raise ValueError(f"negative horizon: {horizon}")
    first = int(realized_rows[0]) if len(realized_rows) else int(pool[0])
    if first - lookback < 0:
        raise ValueError("not enough panel history before the first realized session")
    sim = np.empty((n_worlds, horizon), dtype=np.int64)
    if horizon:
        m = pool.size
        sim[:, 0] = rng.integers(0, m, n_worlds)
        jump = rng.random((n_worlds, horizon)) < 1.0 / float(mean_block)
        fresh = rng.integers(0, m, (n_worlds, horizon))
        for d in range(1, horizon):
            sim[:, d] = np.where(jump[:, d], fresh[:, d], (sim[:, d - 1] + 1) % m)
        sim_rows = pool[sim]
    else:
        sim_rows = np.empty((n_worlds, 0), dtype=np.int64)
    real = np.broadcast_to(np.asarray(list(realized_rows), dtype=np.int64)[None, :], (n_worlds, len(realized_rows)))
    rows = np.concatenate([real, sim_rows], axis=1)
    log_nav0 = np.broadcast_to(panel.log_nav[first - lookback : first][None], (n_worlds, lookback, len(panel.names))).copy()
    return Worlds(rows=rows, log_nav0=log_nav0)


def simulate_contest(
    panel: VehiclePanel,
    worlds: Worlds,
    crowd: CrowdSpec,
    actions: Mapping[str, Callable[[int, np.ndarray], np.ndarray]],
    weights: Mapping[str, Sequence[float]],
    on_day_end: Callable[[int, "SimState"], None] | None,
    seed: int,
) -> SimResult:
    """Run crowd agents and our candidate action paths jointly over every world.

    Each action maps `(day, current holding [W])` to the next holding `[W]`;
    decisions only ever observe closes up to day d-1. Our fills never affect
    the crowd. `weights` holds an optional per-action per-session exposure
    schedule (the gap leg of day d uses the weight of day d-1).

    Returns SimResult with crowd final equity [A, W] (float32) and our final equity per action [W] (float64).
    """
    n_worlds, horizon = worlds.n_worlds, worlds.horizon
    n_agents = crowd.n_agents
    n_vehicles = len(panel.names)
    lookback = worlds.log_nav0.shape[1]
    for name, sched in weights.items():
        if name in actions and len(sched) != horizon:
            raise ValueError(f"weight schedule length mismatch for {name}: {len(sched)} != {horizon}")
    rng = np.random.default_rng(seed)
    ar_w = np.arange(n_worlds)
    buf = np.empty((n_worlds, lookback + horizon, n_vehicles), dtype=np.float32)
    base = worlds.log_nav0[:, -1:, :]
    buf[:, :lookback, :] = (worlds.log_nav0 - base).astype(np.float32)
    held = np.full((n_agents, n_worlds), -1, dtype=np.int16)
    eq = np.ones((n_agents, n_worlds), dtype=np.float32)
    ours_held = {name: np.full(n_worlds, -1, dtype=np.int16) for name in actions}
    ours_eq = {name: np.ones(n_worlds, dtype=np.float64) for name in actions}
    sched_w: dict[str, np.ndarray] = {
        name: (np.asarray(weights[name], dtype=np.float64) if name in weights else np.ones(horizon))
        for name in actions
    }
    wgt = crowd.weight.astype(np.float32)[:, None]
    groups: dict[tuple[int, int, int], list[int]] = {}
    for kind_id in (1, 2, 3):
        for a in np.where(crowd.kind == kind_id)[0]:
            groups.setdefault((kind_id, int(crowd.k[a]), int(crowd.set_id[a])), []).append(int(a))
    grouped = {g: np.array(v) for g, v in groups.items()}
    hold_idx = np.where(crowd.kind == 0)[0]
    gap_t = panel.gap
    intra_t = panel.intraday
    for d in range(horizon):
        t = lookback + d - 1
        new = held.copy()
        if len(hold_idx):
            if crowd.vehicle_per_world is not None:
                veh = crowd.vehicle_per_world
            else:
                veh = np.broadcast_to(crowd.vehicle[hold_idx][:, None], (len(hold_idx), n_worlds))
            active = (d >= crowd.entry[hold_idx])[:, None]
            new[hold_idx] = np.where(active, veh, held[hold_idx])
        for (kind_id, k, sid), members in grouped.items():
            cand = crowd.sets[sid]
            active = (d >= crowd.entry[members])[:, None]
            flip = rng.random((len(members), n_worlds)) < crowd.q[members][:, None]
            flip |= held[members] < 0
            if kind_id in (1, 2):
                mm = np.exp(buf[:, t, :][:, cand] - buf[:, t - k, :][:, cand]) - 1.0
                pick = cand[np.argmax(mm, axis=1)] if kind_id == 1 else cand[np.argmin(mm, axis=1)]
                choice = np.broadcast_to(pick[None, :], (len(members), n_worlds))
            else:
                choice = cand[rng.integers(0, len(cand), (len(members), n_worlds))]
            new[members] = np.where(active & flip, choice, held[members])
        ours_new = {name: np.asarray(fn(d, ours_held[name]), dtype=np.int16) for name, fn in actions.items()}
        day_rows = worlds.rows[:, d]
        g_mat = gap_t[day_rows].T
        o_mat = intra_t[day_rows].T
        hv = np.maximum(held, 0)
        nv = np.maximum(new, 0)
        g = np.where(held >= 0, g_mat[hv, ar_w[None, :]], 0.0)
        o = np.where(new >= 0, o_mat[nv, ar_w[None, :]], 0.0)
        eq *= (1 + wgt * g) * (1 + wgt * o)
        held = new
        for name in actions:
            h0, h1 = ours_held[name], ours_new[name]
            w = sched_w[name][d]
            w0 = sched_w[name][max(d - 1, 0)]
            gg = np.where(h0 >= 0, g_mat[np.maximum(h0, 0), ar_w], 0.0)
            oo = np.where(h1 >= 0, o_mat[np.maximum(h1, 0), ar_w], 0.0)
            ours_eq[name] *= (1 + w0 * gg) * (1 + w * oo)
            ours_held[name] = h1
        cc = (1 + g_mat.T.astype(np.float64)) * (1 + o_mat.T.astype(np.float64)) - 1.0
        buf[:, lookback + d, :] = buf[:, lookback + d - 1, :] + np.log1p(np.clip(cc, -0.999999, None)).astype(np.float32)
        if on_day_end is not None:
            on_day_end(d, SimState(day=d, eq=eq, held=held))
        if d % 5 == 4 or d == horizon - 1:
            logger.info(f"[ALGO] contest_sim progress day={d + 1}/{horizon}")
    return SimResult(crowd_equity=eq, ours=ours_eq)


def rank_metrics(ours: np.ndarray, crowd_equity: np.ndarray) -> dict[str, float]:
    """P1 (no agent strictly above us), P2, P10, median return, P(loss > 30%)."""
    better = (crowd_equity.astype(np.float64) > ours[None, :].astype(np.float64)).sum(axis=0)
    return {
        "P1": float((better == 0).mean()),
        "P2": float((better <= 1).mean()),
        "P10": float((better <= 9).mean()),
        "med_ret": float(np.median(ours) - 1),
        "P_loss30": float((ours < 0.70).mean()),
    }


__all__ = [
    "CrowdSpec",
    "SimResult",
    "SimState",
    "Worlds",
    "bootstrap_worlds",
    "rank_metrics",
    "simulate_contest",
]
