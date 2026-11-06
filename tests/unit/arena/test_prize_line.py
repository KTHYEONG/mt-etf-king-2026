from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.arena import prize_line as prize
from src.arena.evaluation import EvalConfig
from src.arena.panel import Panel, PanelConfig, build_panel
from src.arena.prize_line import (
    PrizeLineConfig,
    load_prize_line_config,
    reference_sigma,
    vol_scaled_line,
    window_stats,
)
from src.core.config import ConfigError

PANEL_CFG = PanelConfig(lock_timeout_s=900.0, nav_divergence_max=0.10)
REF = "REF"
LOOKBACK = 5
SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date(),
    "ticker": pl.String(),
    "name": pl.String(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Int64(),
    "trading_value": pl.Int64(),
    "nav": pl.Float64(),
    "shares_outstanding": pl.Int64(),
    "underlying_index_name": pl.String(),
}
FloatArray = np.ndarray


def _days(n: int) -> list[date]:
    days: list[date] = []
    day = date(2024, 1, 1)
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _panel(returns: FloatArray) -> Panel:
    days = _days(len(returns))
    closes = 100.0 * np.cumprod(1.0 + returns)
    rows = [
        {
            "date": day,
            "ticker": REF,
            "name": "ref fund",
            "open": float(close),
            "high": float(close) * 1.01,
            "low": float(close) * 0.99,
            "close": float(close),
            "volume": 1000,
            "trading_value": 10**9,
            "nav": float(close),
            "shares_outstanding": 1000,
            "underlying_index_name": "IDX",
        }
        for day, close in zip(days, closes, strict=True)
    ]
    return build_panel(pl.DataFrame(rows, schema=SCHEMA), [REF], days[0], days[-1], PANEL_CFG)


def _prize(**changes: Any) -> PrizeLineConfig:
    base = PrizeLineConfig(reference=REF, lookback=LOOKBACK, ks=(1.0,), twin_draws=3, seed=1)
    return replace(base, **changes)


def _eval(**changes: Any) -> EvalConfig:
    base = EvalConfig(
        horizon_sessions=4,
        bootstrap_mean_block=2,
        bootstrap_draws=100,
        seed=1,
        ci_level=0.95,
        exceedance_grid=(1.0,),
        min_sessions=4,
        lock_in=True,
        lock_exit_cost_bps=0.0,
        lock_trigger_buffer=0.0,
        lock_noise_sd=0.0,
        lock_noise_df=4.0,
    )
    return replace(base, **changes)


def _load_with(monkeypatch: pytest.MonkeyPatch, mutate: Callable[[dict[str, Any]], None]) -> PrizeLineConfig:
    section: dict[str, Any] = {
        "reference": "0193T0",
        "lookback": 60,
        "ks": [0.9, 1.3],
        "twin_draws": 20,
        "seed": 20261105,
    }
    mutate(section)
    data: Mapping[str, Any] = {"prize_line_vol": section}
    monkeypatch.setattr(prize, "load_config", lambda name: data)
    return load_prize_line_config()


def test_shipped_config_loads() -> None:
    cfg = load_prize_line_config()
    assert cfg == PrizeLineConfig(reference="0193T0", lookback=60, ks=(0.9, 1.3), twin_draws=20, seed=20261105)


def test_config_loads_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    def _set(**changes: Any) -> Callable[[dict[str, Any]], None]:
        return lambda section: section.update(changes)

    def _drop(key: str) -> Callable[[dict[str, Any]], None]:
        return lambda section: section.pop(key)

    bad_mutations: list[Callable[[dict[str, Any]], None]] = [
        _set(extra=1),
        _drop("seed"),
        _set(ks=[1.3, 0.9]),
        _set(ks=[0.0, 1.0]),
        _set(ks=[]),
        _set(lookback=1),
        _set(twin_draws=0),
        _set(seed="x"),
        _set(reference=""),
        _set(ks="x"),
        _set(ks=[1, True]),
    ]
    for mutate in bad_mutations:
        with pytest.raises(ConfigError):
            _load_with(monkeypatch, mutate)


def test_config_requires_section(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prize, "load_config", lambda name: {"other": {}})
    with pytest.raises(ConfigError, match="prize_line_vol mapping"):
        load_prize_line_config()


def test_window_stats_rejects_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="aligned"):
        window_stats("discovery", np.zeros(10), np.zeros(9), np.full(10, 0.01), _eval(), _prize())


def test_vol_scaled_line_formula() -> None:
    assert vol_scaled_line(0.02, 36, 1.0) == pytest.approx(float(np.exp(0.12)))
    for sigma, horizon, k in ((0.0, 36, 1.0), (0.02, 0, 1.0), (0.02, 36, 0.0)):
        with pytest.raises(ValueError, match="sigma must be"):
            vol_scaled_line(sigma, horizon, k)


def test_reference_sigma_is_causal() -> None:
    rng = np.random.default_rng(10)
    returns = rng.normal(0.0, 0.01, 40)
    t = 25
    corrupted = returns.copy()
    corrupted[t:] = rng.normal(0.0, 0.5, returns.size - t)
    before = reference_sigma(_panel(returns), _prize())
    after = reference_sigma(_panel(corrupted), _prize())
    assert np.array_equal(before[: t + 1], after[: t + 1], equal_nan=True)


def test_reference_sigma_needs_full_lookback() -> None:
    sigma = reference_sigma(_panel(np.random.default_rng(11).normal(0.0, 0.01, 30)), _prize())
    assert np.isnan(sigma[: LOOKBACK + 1]).all()
    assert np.isfinite(sigma[LOOKBACK + 1 :]).all()


def test_reference_sigma_missing_ticker() -> None:
    panel = _panel(np.random.default_rng(12).normal(0.0, 0.01, 20))
    with pytest.raises(ValueError, match="absent"):
        reference_sigma(panel, _prize(reference="OTHER"))


def test_window_stats_none_without_valid_start() -> None:
    net = np.full(10, 0.001)
    assert window_stats("discovery", net, np.zeros(10), np.full(10, np.nan), _eval(), _prize()) is None
    short = np.full(3, 0.001)
    assert window_stats("discovery", short, np.zeros(3), np.full(3, 0.01), _eval(), _prize()) is None


def test_window_stats_lock_dominates_hold_on_rising_then_falling() -> None:
    net = np.array([0.05, 0.0, -0.1, 0.0])
    stats = window_stats("holdout", net, np.zeros(4), np.full(4, 0.01), _eval(), _prize())
    assert stats is not None
    assert stats.n_windows == 1
    assert stats.p_lock == (1.0,)
    assert stats.p_hold == (0.0,)


def test_window_stats_lock_off_equals_hold() -> None:
    rng = np.random.default_rng(13)
    net = rng.normal(0.001, 0.02, 60)
    stats = window_stats("discovery", net, np.full(60, 0.001), np.full(60, 0.02), _eval(lock_in=False), _prize())
    assert stats is not None
    assert stats.p_lock == stats.p_hold


def test_window_stats_twin_deterministic() -> None:
    rng = np.random.default_rng(14)
    net = rng.normal(0.001, 0.02, 60)
    costs = np.full(60, 0.0003)
    sigma = np.full(60, 0.02)
    eval_cfg = _eval(lock_trigger_buffer=0.01, lock_noise_sd=0.0093)
    first = window_stats("discovery", net, costs, sigma, eval_cfg, _prize(ks=(0.9, 1.3)))
    second = window_stats("discovery", net, costs, sigma, eval_cfg, _prize(ks=(0.9, 1.3)))
    assert first == second
    assert first is not None
    assert all(0.0 <= p <= 1.0 for p in first.p_twin_lock)


def test_window_stats_lock_level_scale_moves_the_lock_and_scores_over_uncertainty() -> None:
    net = np.array([0.05, 0.0, -0.1, 0.0])
    unknown = _eval(line_uncertainty=(1.0, 3.0))
    at_line = window_stats("holdout", net, np.zeros(4), np.full(4, 0.01), unknown, _prize())
    above = window_stats("holdout", net, np.zeros(4), np.full(4, 0.01), replace(unknown, lock_level_scale=3.0), _prize())
    assert at_line is not None and above is not None
    assert at_line.p_lock == (0.5,)
    assert above.p_lock == (0.0,)
    assert at_line.p_hold == above.p_hold == (0.0,)
