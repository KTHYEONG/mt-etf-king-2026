"""Invariant guards for the contest simulation engine."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest

from src.contest.engine import (
    CrowdSpec,
    Worlds,
    bootstrap_worlds,
    rank_metrics,
    simulate_contest,
)
from src.contest.panel import VehiclePanel

T = 100
NAMES = ("A", "B", "C")


def _toy_panel() -> VehiclePanel:
    dates = tuple(date(2026, 1, 1) + timedelta(days=i) for i in range(T))
    rng = np.random.default_rng(0)
    gap = (rng.uniform(-0.02, 0.02, (T, 3))).astype(np.float32)
    intra = (rng.uniform(-0.02, 0.02, (T, 3))).astype(np.float32)
    gap[0] = 0.0
    cc = (1 + gap.astype(np.float64)) * (1 + intra.astype(np.float64)) - 1.0
    log_nav = np.cumsum(np.log1p(cc), axis=0)
    return VehiclePanel(dates=dates, names=NAMES, gap=gap, intraday=intra, log_nav=log_nav)


def _toy_worlds(n_worlds: int = 4, horizon: int = 5, seed: int = 7) -> Worlds:
    panel = _toy_panel()
    return bootstrap_worlds(panel, [70, 71], np.arange(0, 70), n_worlds, horizon, 10.0, seed)


def _hold_crowd(n_agents: int, n_worlds: int, vehicle: int = 0) -> CrowdSpec:
    return CrowdSpec(
        kind=np.zeros(n_agents, dtype=np.int8),
        vehicle=np.full(n_agents, vehicle, dtype=np.int16),
        entry=np.zeros(n_agents, dtype=np.int64),
        weight=np.ones(n_agents, dtype=np.float32),
        k=np.zeros(n_agents, dtype=np.int64),
        q=np.zeros(n_agents, dtype=np.float32),
        set_id=np.zeros(n_agents, dtype=np.int64),
        sets=[np.array([0, 1, 2], dtype=np.int16)],
        vehicle_per_world=np.full((n_agents, n_worlds), vehicle, dtype=np.int16),
    )


def test_bootstrap_worlds_shape_and_history() -> None:
    """Realized rows lead every world identically; history precedes day 0."""
    worlds = _toy_worlds(n_worlds=6, horizon=5)
    assert worlds.rows.shape == (6, 7)
    assert worlds.n_worlds == 6
    assert worlds.horizon == 7
    assert (worlds.rows[:, 0] == 70).all() and (worlds.rows[:, 1] == 71).all()
    assert worlds.log_nav0.shape == (6, 61, 3)


def test_bootstrap_worlds_rejects_bad_inputs() -> None:
    """Empty pools, negative horizons, and missing history fail closed."""
    panel = _toy_panel()
    with pytest.raises(ValueError, match=r"empty pool"):
        bootstrap_worlds(panel, [70, 71], np.array([], dtype=np.int64), 2, 3, 10.0, 0)
    with pytest.raises(ValueError, match=r"negative horizon"):
        bootstrap_worlds(panel, [70, 71], np.arange(0, 70), 2, -1, 10.0, 0)
    with pytest.raises(ValueError, match=r"not enough panel history"):
        bootstrap_worlds(panel, [10, 11], np.arange(0, 70), 2, 3, 10.0, 0)


def test_simulate_switch_earns_old_gap_then_new_intraday() -> None:
    """A day-1 switch compounds the old holding's gap with the new holding's intraday leg."""
    dates = tuple(date(2026, 1, 1) + timedelta(days=i) for i in range(T))
    gap = np.zeros((T, 3), dtype=np.float32)
    intra = np.zeros((T, 3), dtype=np.float32)
    gap[:, 0] = 0.01
    intra[:, 0] = 0.02
    gap[:, 1] = 0.03
    intra[:, 1] = 0.04
    cc = (1 + gap.astype(np.float64)) * (1 + intra.astype(np.float64)) - 1.0
    panel = VehiclePanel(dates=dates, names=NAMES, gap=gap, intraday=intra, log_nav=np.cumsum(np.log1p(cc), axis=0))
    worlds = Worlds(rows=np.array([[70, 71]]), log_nav0=np.zeros((1, 61, 3)))
    crowd = _hold_crowd(2, 1)
    actions = {"switch": lambda d, h: np.full_like(h, 0 if d < 1 else 1)}
    out = simulate_contest(panel, worlds, crowd, actions, {}, None, seed=0)
    ours = out.ours["switch"]
    assert ours[0] == pytest.approx((1 + 0.0) * (1 + 0.02) * (1 + 0.01) * (1 + 0.04))
    assert out.crowd_equity.shape == (2, 1)
    assert out.crowd_equity.dtype == np.float32
    assert ours.dtype == np.float64


def test_simulate_no_lookahead_in_decisions() -> None:
    """Perturbing legs drawn on sessions >= d leaves decisions at d unchanged."""
    panel = _toy_panel()
    worlds = Worlds(rows=np.array([[70, 71, 5, 6]]), log_nav0=np.zeros((1, 61, 3)))
    crowd = CrowdSpec(
        kind=np.array([1], dtype=np.int8),
        vehicle=np.zeros(1, dtype=np.int16),
        entry=np.zeros(1, dtype=np.int64),
        weight=np.ones(1, dtype=np.float32),
        k=np.array([3], dtype=np.int64),
        q=np.ones(1, dtype=np.float32),
        set_id=np.zeros(1, dtype=np.int64),
        sets=[np.array([0, 1, 2], dtype=np.int16)],
    )
    first_held: list[np.ndarray] = []
    out1 = simulate_contest(
        panel, worlds, crowd, {}, {}, lambda d, s: first_held.append(s.held.copy()), seed=1
    )
    mask = np.isin(np.arange(T), [5, 6])[:, None]
    panel2 = VehiclePanel(
        dates=panel.dates,
        names=panel.names,
        gap=np.where(mask, panel.gap + 0.05, panel.gap).astype(np.float32),
        intraday=panel.intraday,
        log_nav=panel.log_nav,
    )
    second_held: list[np.ndarray] = []
    out2 = simulate_contest(
        panel2, worlds, crowd, {}, {}, lambda d, s: second_held.append(s.held.copy()), seed=1
    )
    assert len(first_held) == len(second_held) == 4
    for day in range(3):
        assert np.array_equal(first_held[day], second_held[day])
    assert not np.allclose(out1.crowd_equity, out2.crowd_equity)


def test_simulate_mixed_behaviours_and_cash_fallback() -> None:
    """Momentum, contrarian, and random groups update; holds fall back to the vehicle field."""
    panel = _toy_panel()
    worlds = _toy_worlds(n_worlds=2, horizon=4)
    crowd = CrowdSpec(
        kind=np.array([1, 2, 3, 0], dtype=np.int8),
        vehicle=np.array([0, 0, 0, 1], dtype=np.int16),
        entry=np.zeros(4, dtype=np.int64),
        weight=np.ones(4, dtype=np.float32),
        k=np.array([3, 5, 0, 0], dtype=np.int64),
        q=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        set_id=np.array([0, 0, 0, 0], dtype=np.int64),
        sets=[np.array([0, 1, 2], dtype=np.int16)],
        vehicle_per_world=None,
    )
    out = simulate_contest(panel, worlds, crowd, {}, {}, None, seed=2)
    assert out.crowd_equity.shape == (4, 2)
    assert (out.crowd_equity != 1.0).all()


def test_bootstrap_zero_horizon_keeps_realized_only() -> None:
    """A zero bootstrap horizon yields the realized rows with full history."""
    panel = _toy_panel()
    worlds = bootstrap_worlds(panel, [70, 71], np.arange(0, 70), 3, 0, 10.0, 0)
    assert worlds.rows.shape == (3, 2)
    assert (worlds.rows[:, 0] == 70).all() and (worlds.rows[:, 1] == 71).all()
    assert worlds.log_nav0.shape == (3, 61, 3)
    out = simulate_contest(panel, worlds, _hold_crowd(2, 3), {}, {}, None, seed=0)
    g = panel.gap[[70, 71], 0].astype(np.float64)
    o = panel.intraday[[70, 71], 0].astype(np.float64)
    assert np.allclose(out.crowd_equity, float((1 + o[0]) * (1 + g[1]) * (1 + o[1])))


def test_simulate_hold_agents_in_cash_before_entry() -> None:
    """Agents hold cash (equity 1.0) until their entry day."""
    panel = _toy_panel()
    worlds = _toy_worlds(n_worlds=2, horizon=3)
    crowd = _hold_crowd(2, 2)
    crowd.entry[:] = 2
    snapshots: list[np.ndarray] = []
    out = simulate_contest(panel, worlds, crowd, {}, {}, lambda d, s: snapshots.append(s.eq.copy()), seed=0)
    assert np.all(snapshots[0] == 1.0)
    assert np.all(snapshots[1] == 1.0)
    assert np.all(out.crowd_equity != 1.0)
    assert out.ours == {}


def test_rank_metrics_on_ties() -> None:
    """Equity equal to the best agent counts as a win under the strictly-greater rule."""
    crowd = np.array([[1.0, 2.0], [1.5, 2.0], [0.5, 3.0]])
    ours = np.array([1.5, 2.0])
    m = rank_metrics(ours, crowd)
    assert m["P1"] == pytest.approx(0.5)
    assert m["P2"] == pytest.approx(1.0)
    assert m["P10"] == pytest.approx(1.0)
    assert m["med_ret"] == pytest.approx(0.75)
    assert m["P_loss30"] == pytest.approx(0.0)


def test_simulate_deterministic_under_seed() -> None:
    """Identical inputs and seeds give identical outputs."""
    panel = _toy_panel()
    crowd = _hold_crowd(5, 3)
    worlds = _toy_worlds(n_worlds=3, horizon=6)
    actions: dict[str, Any] = {"hold_a": lambda d, h: np.full_like(h, 0)}
    first = simulate_contest(panel, worlds, crowd, actions, {"hold_a": [0.9] * 8}, None, seed=3)
    second = simulate_contest(panel, worlds, crowd, actions, {"hold_a": [0.9] * 8}, None, seed=3)
    assert np.array_equal(first.crowd_equity, second.crowd_equity)
    assert np.array_equal(first.ours["hold_a"], second.ours["hold_a"])


def test_simulate_weight_schedule_length_mismatch() -> None:
    """A weight schedule that does not span the horizon fails closed."""
    panel = _toy_panel()
    worlds = _toy_worlds(n_worlds=2, horizon=3)
    crowd = _hold_crowd(2, 2)
    actions: dict[str, Any] = {"hold_a": lambda d, h: np.full_like(h, 0)}
    with pytest.raises(ValueError, match=r"weight schedule length"):
        simulate_contest(panel, worlds, crowd, actions, {"hold_a": [1.0, 1.0]}, None, seed=0)
