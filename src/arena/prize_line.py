"""Descriptive vol-scaled prize line on real rolling windows: causal reference sigma and lock-in probabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from math import exp, sqrt
from typing import Any, Final

import numpy as np
import polars as pl
from numpy import typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from src.arena.evaluation import EvalConfig, lock_in_multiples, pre_auction_quotes
from src.arena.panel import Panel
from src.core.config import ConfigError, load_config

FloatArray = npt.NDArray[np.float64]
_PRIZE_KEYS: Final = frozenset({"reference", "lookback", "ks", "twin_draws", "seed"})
_BPS_PER_UNIT: Final = 10_000.0
_MIN_LOOKBACK: Final = 2


@dataclass(frozen=True)
class PrizeLineConfig:
    reference: str
    lookback: int
    ks: tuple[float, ...]
    twin_draws: int
    seed: int


@dataclass(frozen=True)
class VolWindowStats:
    period: str
    n_windows: int
    n_effective: int
    ks: tuple[float, ...]
    median_line: tuple[float, ...]
    p_lock: tuple[float, ...]
    p_hold: tuple[float, ...]
    p_twin_lock: tuple[float, ...]


def _int(value: Any, key: str) -> int:
    if type(value) is not int:
        raise ConfigError(f"prize_line_vol.{key} must be an integer")
    return value


def load_prize_line_config(name: str = "arena") -> PrizeLineConfig:
    """Load and validate the `prize_line_vol` mapping of `configs/<name>.yaml`; raises ConfigError on any problem."""
    section = load_config(name).get("prize_line_vol")
    if not isinstance(section, Mapping):
        raise ConfigError(f"config {name} must define a prize_line_vol mapping")
    if set(section) != _PRIZE_KEYS:
        raise ConfigError(f"prize_line_vol keys must be exactly {sorted(_PRIZE_KEYS)}")
    reference = section["reference"]
    if not isinstance(reference, str) or not reference:
        raise ConfigError("prize_line_vol.reference must be a non-empty string")
    raw_ks = section["ks"]
    if not isinstance(raw_ks, list) or not raw_ks:
        raise ConfigError("prize_line_vol.ks must be a non-empty list")
    if not all(isinstance(k, (int, float)) and not isinstance(k, bool) for k in raw_ks):
        raise ConfigError("prize_line_vol.ks must contain numbers")
    cfg = PrizeLineConfig(
        reference=reference,
        lookback=_int(section["lookback"], "lookback"),
        ks=tuple(float(k) for k in raw_ks),
        twin_draws=_int(section["twin_draws"], "twin_draws"),
        seed=_int(section["seed"], "seed"),
    )
    if cfg.lookback < _MIN_LOOKBACK:
        raise ConfigError(f"prize_line_vol.lookback must be >= {_MIN_LOOKBACK}")
    if not all(k > 0.0 for k in cfg.ks) or not all(b > a for a, b in pairwise(cfg.ks)):
        raise ConfigError("prize_line_vol.ks must be strictly increasing and > 0")
    if cfg.twin_draws < 1:
        raise ConfigError("prize_line_vol.twin_draws must be >= 1")
    return cfg


def vol_scaled_line(sigma: float, horizon: int, k: float) -> float:
    """Return the prize line `exp(k * sigma * sqrt(horizon))`; raises ValueError unless sigma > 0, horizon >= 1 and k > 0."""
    if not sigma > 0.0 or horizon < 1 or not k > 0.0:
        raise ValueError("sigma must be > 0, horizon must be >= 1 and k must be > 0")
    return float(exp(k * sigma * sqrt(horizon)))


def reference_sigma(panel: Panel, cfg: PrizeLineConfig) -> FloatArray:
    """Return the causal daily-return sd of the reference per session; NaN unless all `lookback` prior returns are finite."""
    reference = panel.frame.filter(pl.col("ticker") == cfg.reference)
    if reference.is_empty():
        raise ValueError(f"reference ticker {cfg.reference} is absent from the panel")
    aligned = pl.DataFrame({"date": list(panel.sessions)}, schema={"date": pl.Date}).join(
        reference.select("date", "r_ref"), on="date", how="left"
    )
    returns = aligned["r_ref"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    lookback = cfg.lookback
    padded = np.concatenate([np.full(lookback, np.nan), returns])
    past = sliding_window_view(padded, lookback)[: len(returns)]
    finite = np.isfinite(past)
    complete = finite.all(axis=1)
    mean = np.where(finite, past, 0.0).mean(axis=1)
    squares = np.where(finite, (past - mean[:, None]) ** 2, 0.0).sum(axis=1)
    return np.where(complete, np.sqrt(squares / (lookback - 1)), np.nan)


def window_stats(
    period: str,
    net: FloatArray,
    costs: FloatArray,
    sigma: FloatArray,
    eval_cfg: EvalConfig,
    cfg: PrizeLineConfig,
) -> VolWindowStats | None:
    """Return descriptive prize-line statistics over the valid windows of one period, or None when no start is valid."""
    horizon = eval_cfg.horizon_sessions
    if not net.shape == costs.shape == sigma.shape or net.ndim != 1:
        raise ValueError("net, costs and sigma must be aligned 1-D arrays")
    count = len(net) - horizon + 1
    if count < 1:
        return None
    starts = np.flatnonzero(np.isfinite(sigma[:count]) & (sigma[:count] > 0.0))
    if starts.size == 0:
        return None
    windows = starts[:, None] + np.arange(horizon)[None, :]
    gross = np.abs(net + costs)
    lines = [np.array([vol_scaled_line(s, horizon, k) for s in sigma[starts]]) for k in cfg.ks]
    lock_on = eval_cfg.lock_in
    exit_cost = eval_cfg.lock_exit_cost_bps / _BPS_PER_UNIT
    buffer = eval_cfg.lock_trigger_buffer
    noise = (eval_cfg.lock_noise_sd, eval_cfg.lock_noise_df)

    quote_seed, sign_seed = np.random.SeedSequence(cfg.seed).spawn(2)
    quote_rng = np.random.default_rng(quote_seed)
    paths = np.cumprod(1.0 + np.maximum(net[windows], -1.0), axis=1)
    quotes = pre_auction_quotes(paths, *noise, quote_rng) if lock_on else paths

    def terminal(path: FloatArray, quote: FloatArray, line: FloatArray) -> FloatArray:
        if not lock_on:
            return path[:, -1]
        return lock_in_multiples(path, quote, line, exit_cost, buffer)

    p_hold = tuple(float(np.mean(paths[:, -1] >= line)) for line in lines)
    p_lock = tuple(float(np.mean(terminal(paths, quotes, line) >= line)) for line in lines)

    sign_rng = np.random.default_rng(sign_seed)
    signs = sign_rng.choice(np.array([-1, 1], dtype=np.int8), size=(cfg.twin_draws, starts.size, horizon))
    twin_lock = np.zeros(len(lines))
    for draw in range(cfg.twin_draws):
        r0 = np.maximum(signs[draw] * gross[windows] - costs[windows], -1.0)
        twin_paths = np.cumprod(1.0 + r0, axis=1)
        twin_quotes = pre_auction_quotes(twin_paths, *noise, quote_rng) if lock_on else twin_paths
        for j, line in enumerate(lines):
            twin_lock[j] += float(np.mean(terminal(twin_paths, twin_quotes, line) >= line))
    twin_lock /= cfg.twin_draws

    return VolWindowStats(
        period=period,
        n_windows=int(starts.size),
        n_effective=int(starts.size) // horizon,
        ks=cfg.ks,
        median_line=tuple(float(np.median(line)) for line in lines),
        p_lock=p_lock,
        p_hold=p_hold,
        p_twin_lock=tuple(float(p) for p in twin_lock),
    )
