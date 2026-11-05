import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from src.arena import evaluation as ev
from src.arena.evaluation import (
    DegenerateSeriesError,
    EvalConfig,
    EvaluationReport,
    InsufficientDataError,
    edge_stats,
    evaluate,
    exceedance,
    lock_in_multiples,
    load_eval_config,
    report_to_dict,
    stationary_bootstrap_indices,
)
from src.core.config import ConfigError, load_config

_BASE_EVALUATION: dict[str, Any] = dict(load_config("arena")["evaluation"])

CFG = EvalConfig(
    horizon_sessions=20,
    bootstrap_mean_block=10,
    bootstrap_draws=400,
    seed=7,
    ci_level=0.95,
    exceedance_grid=(1.1, 1.3, 1.5, 2.0, 2.3, 3.0),
    min_sessions=60,
)
START = date(2024, 1, 1)
FloatArray = npt.NDArray[np.float64]


def _sessions(n: int) -> list[date]:
    return [START + timedelta(days=i) for i in range(n)]


def _series(n: int, mu: float, sigma: float, seed: int) -> tuple[FloatArray, FloatArray]:
    costs = np.full(n, 0.001)
    net = np.random.default_rng(seed).normal(mu, sigma, n) - costs
    return net, costs


def _report(net: FloatArray, costs: FloatArray, cfg: EvalConfig = CFG) -> EvaluationReport:
    return evaluate("s1", "h1", _sessions(len(net)), net, costs, cfg, 1)


def _load_with(monkeypatch: pytest.MonkeyPatch, mutate: Callable[[dict[str, Any]], None]) -> EvalConfig:
    section: dict[str, Any] = dict(_BASE_EVALUATION)
    mutate(section)
    data: Mapping[str, Any] = {"evaluation": section}
    monkeypatch.setattr(ev, "load_config", lambda name: data)
    return load_eval_config()


def test_config_loads_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_eval_config() == EvalConfig(
        36, 10, 5000, 20261104, 0.95, (1.1, 1.3, 1.5, 2.0, 2.3, 3.0), 252, True, 3.0
    )

    def _set(**changes: Any) -> Callable[[dict[str, Any]], None]:
        return lambda section: section.update(changes)

    def _drop(key: str) -> Callable[[dict[str, Any]], None]:
        return lambda section: section.pop(key)

    bad_mutations: list[Callable[[dict[str, Any]], None]] = [
        _set(extra=1),
        _drop("seed"),
        _set(bootstrap_draws=True),
        _set(seed="x"),
        _set(exceedance_grid=[1.3, 1.1]),
        _set(exceedance_grid=[]),
        _set(exceedance_grid=[1.1, True]),
        _set(horizon_sessions=0),
        _set(bootstrap_mean_block=0),
        _set(bootstrap_draws=99),
        _set(ci_level=1.0),
        _set(exceedance_grid=[0.0, 1.0]),
        _set(min_sessions=5),
        _drop("lock_in"),
        _set(lock_in=1),
        _set(lock_exit_cost_bps=-1.0),
        _set(lock_exit_cost_bps=10_000.0),
    ]
    for mutate in bad_mutations:
        with pytest.raises(ConfigError):
            _load_with(monkeypatch, mutate)

    monkeypatch.setattr(ev, "load_config", lambda name: {"other": {}})
    with pytest.raises(ConfigError):
        load_eval_config()
    monkeypatch.setattr(ev, "load_config", lambda name: {"evaluation": []})
    with pytest.raises(ConfigError):
        load_eval_config()


def test_bootstrap_indices_shape_range_and_continuity() -> None:
    n, length, block, draws = 50, 30, 5, 400
    idx = stationary_bootstrap_indices(n, length, block, draws, np.random.default_rng(1))
    assert idx.shape == (draws, length)
    assert idx.min() >= 0 and idx.max() < n
    breaks = idx[:, 1:] != (idx[:, :-1] + 1) % n
    mean_run = idx.size / (int(breaks.sum()) + draws)
    assert 0.8 * block <= mean_run <= 1.2 * block


def test_bootstrap_rejects_bad_args() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="must be >= 1"):
        stationary_bootstrap_indices(0, 5, 5, 10, rng)
    with pytest.raises(ValueError, match="must be >= 1"):
        stationary_bootstrap_indices(5, 5, 0, 10, rng)


def test_deterministic_report() -> None:
    net, costs = _series(120, 0.003, 0.02, 3)
    assert report_to_dict(_report(net, costs)) == report_to_dict(_report(net, costs))


def test_timing_edge_detected() -> None:
    gross = np.random.default_rng(11).normal(0.003, 0.02, 1000)
    rep = _report(gross, np.zeros(1000))
    assert rep.edge.ci_low > 0
    assert rep.edge.p_value < 0.05
    assert rep.exceedance.lift[0] > 0


def test_no_edge_not_detected() -> None:
    gross = np.random.default_rng(12).normal(0.0, 0.02, 1000)
    gross = gross - gross.mean()
    rep = _report(gross, np.zeros(1000))
    assert rep.edge.mean_gross == pytest.approx(0.0, abs=1e-15)
    assert rep.edge.p_value >= 0.4


def test_haircut_monotone_in_trials() -> None:
    gross = np.random.default_rng(13).normal(0.001, 0.02, 300)
    idx = stationary_bootstrap_indices(300, 300, 10, CFG.bootstrap_draws, np.random.default_rng(2))
    base = edge_stats(gross, idx, CFG, 1)
    adjusted = [edge_stats(gross, idx, CFG, k).p_value_adjusted for k in (1, 10, 100)]
    assert adjusted[0] == pytest.approx(base.p_value)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
    with pytest.raises(ValueError, match="n_trials must be"):
        edge_stats(gross, idx, CFG, 0)


def test_degenerate_constant_series() -> None:
    idx = stationary_bootstrap_indices(100, 100, 10, CFG.bootstrap_draws, np.random.default_rng(5))
    with pytest.raises(DegenerateSeriesError):
        edge_stats(np.full(100, 0.001), idx, CFG, 1)
    with pytest.raises(DegenerateSeriesError):
        _report(np.full(100, 0.001), np.zeros(100))


def test_twin_ignores_direction() -> None:
    n = 200
    g = np.random.default_rng(14).normal(0.002, 0.03, n)
    costs = np.full(n, 0.001)
    idx = stationary_bootstrap_indices(n, CFG.horizon_sessions, 10, CFG.bootstrap_draws, np.random.default_rng(3))
    signs = np.random.default_rng(4).choice(np.array([-1, 1], dtype=np.int8), size=idx.shape)
    up = exceedance(g - costs, costs, idx, signs, CFG)
    down = exceedance(-g - costs, costs, idx, signs, CFG)
    assert up.p_twin == down.p_twin
    assert up.twin_q50 == down.twin_q50
    assert up.p_strategy != down.p_strategy


def test_exceedance_monotone_in_threshold() -> None:
    net, costs = _series(200, 0.002, 0.03, 5)
    ex = _report(net, costs).exceedance
    assert all(a >= b for a, b in zip(ex.p_strategy, ex.p_strategy[1:], strict=False))
    assert all(a >= b for a, b in zip(ex.p_twin, ex.p_twin[1:], strict=False))
    for s, t, lift in zip(ex.p_strategy, ex.p_twin, ex.lift, strict=True):
        assert lift == pytest.approx(s - t, abs=0.0)
    assert ex.q05 <= ex.q50 <= ex.q95


def test_costs_lower_both_curves() -> None:
    gross = np.random.default_rng(6).normal(0.002, 0.03, 200)
    idx = stationary_bootstrap_indices(200, CFG.horizon_sessions, 10, CFG.bootstrap_draws, np.random.default_rng(8))
    signs = np.random.default_rng(9).choice(np.array([-1, 1], dtype=np.int8), size=idx.shape)
    zero = exceedance(gross, np.zeros(200), idx, signs, CFG)
    costly = exceedance(gross - 0.002, np.full(200, 0.002), idx, signs, CFG)
    assert all(c <= z for c, z in zip(costly.p_strategy, zero.p_strategy, strict=True))
    assert all(c <= z for c, z in zip(costly.p_twin, zero.p_twin, strict=True))


def test_wipeout_floor() -> None:
    net = np.full(40, 0.01)
    net[0] = -1.5
    costs = np.zeros(40)
    idx = np.zeros((4, 3), dtype=np.int64)
    neg = exceedance(net, costs, idx, np.full((4, 3), -1, dtype=np.int8), CFG)
    pos = exceedance(net, costs, idx, np.full((4, 3), 1, dtype=np.int8), CFG)
    assert neg.q50 == 0.0
    assert neg.twin_q50 == 0.0
    assert pos.twin_q50 == pytest.approx(2.5**3)
    assert all(np.isfinite(v) for v in neg.p_twin + pos.p_twin)


def test_n_effective_is_non_overlapping() -> None:
    net, costs = _series(1000, 0.001, 0.02, 15)
    assert _report(net, costs).exceedance.n_effective == 50


def test_rejects_bad_inputs() -> None:
    net, costs = _series(100, 0.001, 0.02, 16)
    sessions = _sessions(100)
    bad_net = net.copy()
    bad_net[3] = np.nan
    low = net.copy()
    low[0] = -1.0
    with pytest.raises(ValueError, match="finite"):
        evaluate("s", "h", sessions, bad_net, costs, CFG, 1)
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate("s", "h", sessions[:-1], net, costs, CFG, 1)
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate("s", "h", sessions[::-1], net, costs, CFG, 1)
    with pytest.raises(ValueError, match="must be > -1"):
        evaluate("s", "h", sessions, low, costs, CFG, 1)
    with pytest.raises(ValueError, match="must be >= 0"):
        evaluate("s", "h", sessions, net, -np.ones(100), CFG, 1)


def test_insufficient_history() -> None:
    net, costs = _series(30, 0.001, 0.02, 17)
    with pytest.raises(InsufficientDataError):
        _report(net, costs)


def test_by_year_partition() -> None:
    n = 3 * 260
    sessions = [date(2022, 1, 1) + timedelta(days=i) for i in range(n)]
    net = np.random.default_rng(18).normal(0.0005, 0.02, n)
    rep = evaluate("s", "h", sessions, net, np.zeros(n), CFG, 1)
    years = [y.year for y in rep.by_year]
    assert years == sorted(years)
    assert sum(y.n_sessions for y in rep.by_year) == n
    positive = sum(1 for y in rep.by_year if y.mean_gross > 0)
    assert rep.positive_year_share == pytest.approx(positive / len(rep.by_year))


def test_causal_invariance_of_prefix() -> None:
    net, costs = _series(200, 0.001, 0.02, 19)
    k = 120
    tail_changed = net.copy()
    tail_changed[k:] = np.random.default_rng(20).normal(0.05, 0.2, 200 - k)
    prefix_a = evaluate("s", "h", _sessions(k), net[:k], costs[:k], CFG, 1)
    prefix_b = evaluate("s", "h", _sessions(k), tail_changed[:k], costs[:k], CFG, 1)
    assert report_to_dict(prefix_a) == report_to_dict(prefix_b)


def test_json_safe() -> None:
    net, costs = _series(120, 0.002, 0.02, 21)
    json.dumps(report_to_dict(_report(net, costs)), allow_nan=False)


def test_json_has_hold_fields() -> None:
    net, costs = _series(120, 0.002, 0.02, 22)
    exc = report_to_dict(_report(net, costs))["exceedance"]
    assert {"p_strategy_hold", "p_twin_hold", "lock_in"} <= set(exc)


def test_lock_in_no_cross_unchanged() -> None:
    paths = np.array([[0.9, 0.95, 0.97, 0.99, 1.0, 1.02]])
    assert np.array_equal(lock_in_multiples(paths, 1.1, 0.001), paths[:, -1])


def test_lock_in_exit_next_session_with_cost() -> None:
    paths = np.array([[1.0, 1.1, 1.2, 1.25, 1.1, 1.05]])
    result = lock_in_multiples(paths, 1.2, 0.001)
    assert result[0] == pytest.approx(paths[0, 3] * 0.999)


def test_lock_in_late_cross_no_cost() -> None:
    paths = np.array([[1.0, 1.0, 1.0, 1.2, 1.3], [1.0, 1.0, 1.0, 1.0, 1.3]])
    paths[0, 3] = 1.2
    result = lock_in_multiples(paths, 1.2, 0.5)
    assert np.array_equal(result, paths[:, -1])


def test_lock_in_rejects_bad_args() -> None:
    paths = np.ones((2, 4))
    with pytest.raises(ValueError, match="2-D"):
        lock_in_multiples(np.ones(4), 1.1, 0.0)
    with pytest.raises(ValueError, match="threshold must be > 0"):
        lock_in_multiples(paths, 0.0, 0.0)
    with pytest.raises(ValueError, match="exit_cost must be in"):
        lock_in_multiples(paths, 1.1, 1.0)
    with pytest.raises(ValueError, match="exit_cost must be in"):
        lock_in_multiples(paths, 1.1, -0.1)


def test_lock_in_does_not_mutate() -> None:
    paths = np.array([[1.0, 1.2, 1.3, 1.1], [0.9, 0.95, 1.0, 1.05]])
    before = paths.copy()
    lock_in_multiples(paths, 1.2, 0.001)
    assert np.array_equal(paths, before)


def _lock_draws() -> tuple[FloatArray, FloatArray, npt.NDArray[np.int64], npt.NDArray[np.int8]]:
    n = 200
    net, costs = _series(n, 0.002, 0.03, 23)
    idx = stationary_bootstrap_indices(n, CFG.horizon_sessions, 10, CFG.bootstrap_draws, np.random.default_rng(24))
    signs = np.random.default_rng(25).choice(np.array([-1, 1], dtype=np.int8), size=idx.shape)
    return net, costs, idx, signs


def test_exceedance_lock_off_matches_hold() -> None:
    net, costs, idx, signs = _lock_draws()
    ex = exceedance(net, costs, idx, signs, CFG)
    assert ex.lock_in is False
    assert ex.p_strategy == ex.p_strategy_hold
    assert ex.p_twin == ex.p_twin_hold


def test_exceedance_lock_on_probabilities_and_lift() -> None:
    net, costs, idx, signs = _lock_draws()
    cfg = replace(CFG, lock_in=True, lock_exit_cost_bps=3.0)
    ex = exceedance(net, costs, idx, signs, cfg)
    assert ex.lock_in is True
    assert all(0.0 <= p <= 1.0 for p in ex.p_strategy + ex.p_twin)
    assert ex.lift == tuple(s - t for s, t in zip(ex.p_strategy, ex.p_twin, strict=True))
    assert ex.p_strategy_hold == exceedance(net, costs, idx, signs, CFG).p_strategy


def test_exceedance_lock_on_certain_for_monotone_growth() -> None:
    n = 60
    net = np.full(n, 0.02)
    costs = np.zeros(n)
    idx = np.zeros((4, CFG.horizon_sessions), dtype=np.int64)
    signs = np.ones((4, CFG.horizon_sessions), dtype=np.int8)
    ex = exceedance(net, costs, idx, signs, replace(CFG, lock_in=True, lock_exit_cost_bps=0.0))
    reached = [k for k, w in enumerate(ex.thresholds) if w <= 1.02**5]
    assert reached
    assert all(ex.p_strategy[k] == 1.0 for k in reached)


def test_exceedance_lock_quantiles_unchanged() -> None:
    net, costs, idx, signs = _lock_draws()
    off = exceedance(net, costs, idx, signs, CFG)
    on = exceedance(net, costs, idx, signs, replace(CFG, lock_in=True, lock_exit_cost_bps=3.0))
    assert (off.q05, off.q50, off.q95, off.twin_q50) == (on.q05, on.q50, on.q95, on.twin_q50)
