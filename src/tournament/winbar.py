"""Championship win-bar objective (research-only diagnostic, never a production gate).

The win bar is an ex-post cross-sectional benchmark computed from the same
entry/exit opens as the strategy window. It is carried as a rank/ratio
scenario pair calibrated on the 2025 observation
(winner = cross-sectional rank 5 of 563, 0.621 of the executable oracle),
never as a point estimate and never as an activation signal.
"""

from __future__ import annotations

import heapq
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

WIN_BAR_IS_PRODUCTION_GATE: Final[bool] = False
WIN_BAR_RANK: Final[int] = 5
WIN_BAR_ORACLE_RATIO: Final[float] = 0.621


@dataclass(frozen=True, slots=True)
class WindowWinBar:
    window_start: date
    breadth: int
    bar_rank: float | None
    bar_ratio: float | None


def _validate_bar_params(rank: int, oracle_ratio: float) -> tuple[int, float]:
    # Fail-closed argument validation: no silent default substitution.
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError(f"rank must be an integer >= 1, got {rank!r}")
    if (
        isinstance(oracle_ratio, bool)
        or not isinstance(oracle_ratio, (int, float))
        or not math.isfinite(float(oracle_ratio))
        or not 0.0 < float(oracle_ratio) <= 1.0
    ):
        raise ValueError(f"oracle_ratio must be a finite float in (0.0, 1.0], got {oracle_ratio!r}")
    return int(rank), float(oracle_ratio)


def _valid_open(value: float | None) -> float | None:
    # Price validity: only finite, strictly positive opens contribute; anything
    # else (non-finite, zero, negative, missing) is excluded, never imputed.
    if value is None or isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def window_win_bars(
    open_by_session: Sequence[Mapping[str, float]],
    sessions: Sequence[date],
    candidates_by_session: Mapping[date, Sequence[str]],
    horizon: int,
    *,
    rank: int = WIN_BAR_RANK,
    oracle_ratio: float = WIN_BAR_ORACLE_RATIO,
) -> tuple[WindowWinBar, ...]:
    rank_int, ratio = _validate_bar_params(rank, oracle_ratio)
    # Horizon guard: degenerate grids are a no-op (empty tuple), never an error.
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, (int, float))
        or not float(horizon).is_integer()
        or int(horizon) < 1
        or len(sessions) == 0
    ):
        return ()
    h = int(horizon)
    n = len(sessions)
    out: list[WindowWinBar] = []
    # One linear pass over sessions reusing the already-materialised maps.
    for i in range(n):
        entry = i + 1
        exit_ = entry + h
        if exit_ >= len(open_by_session) or exit_ >= n:
            continue
        tickers = candidates_by_session.get(sessions[i], ())
        entry_map = open_by_session[entry]
        exit_map = open_by_session[exit_]
        rets: list[float] = []
        for ticker in tickers:
            pe = _valid_open(entry_map.get(ticker, None))
            px = _valid_open(exit_map.get(ticker, None))
            if pe is None or px is None:
                continue
            rets.append(float(px) / float(pe) - 1.0)
        # Partial sort of at most `rank` elements per window (float64).
        top_desc = heapq.nlargest(rank_int, rets) if rets else []
        bar_rank = float(top_desc[-1]) if len(top_desc) >= rank_int else None
        bar_ratio = float(max(rets) * ratio) if rets else None
        out.append(
            WindowWinBar(
                window_start=sessions[entry],
                breadth=len(rets),
                bar_rank=bar_rank,
                bar_ratio=bar_ratio,
            )
        )
    return tuple(out)


def win_bar_exceedance(
    window_returns: Sequence[float],
    bars: Sequence[WindowWinBar],
    *,
    model: str,
    min_bar_windows: int = 1,
) -> tuple[float | None, int]:
    # Fail-closed model selector: only the scenario-pair members exist.
    if model != "rank" and model != "ratio":
        raise ValueError(f"model must be exactly 'rank' or 'ratio', got {model!r}")
    # Evidence floor: thin samples withhold the rate instead of guessing.
    need = max(1, int(min_bar_windows))
    hits = 0
    total = 0
    for i, bar in enumerate(bars):
        target = bar.bar_rank if model == "rank" else bar.bar_ratio
        if target is None:
            continue
        total += 1
        if float(window_returns[i]) > float(target):
            hits += 1
    if total < need:
        return (None, int(total))
    return (float(hits) / float(total), int(total))


def build_winbar_summary(
    *,
    sessions: Sequence[date],
    open_map: Mapping[date, Mapping[str, float]],
    candidates_by_session: Mapping[date, Sequence[str]],
    window_returns: Sequence[float],
    horizon: int,
    rank: int = WIN_BAR_RANK,
    oracle_ratio: float = WIN_BAR_ORACLE_RATIO,
    min_bar_windows: int = 30,
) -> dict[str, object]:
    rank_int, ratio = _validate_bar_params(rank, oracle_ratio)
    open_by_session = [dict(open_map.get(s, {})) for s in sessions]
    bars = window_win_bars(
        open_by_session,
        sessions,
        candidates_by_session,
        horizon,
        rank=rank_int,
        oracle_ratio=ratio,
    )
    # Session-aligned returns mirroring build_attainability_summary indexing.
    aligned = [float(window_returns[j]) if j < len(window_returns) else 0.0 for j in range(len(bars))]
    rank_rate, rank_n = win_bar_exceedance(aligned, bars, model="rank", min_bar_windows=min_bar_windows)
    ratio_rate, ratio_n = win_bar_exceedance(aligned, bars, model="ratio", min_bar_windows=min_bar_windows)
    rank_bars = [float(b.bar_rank) for b in bars if b.bar_rank is not None]
    ratio_bars = [float(b.bar_ratio) for b in bars if b.bar_ratio is not None]
    # Medians are descriptive and never gated by the evidence floor.
    median_rank = float(statistics.median(rank_bars)) if rank_bars else None
    median_ratio = float(statistics.median(ratio_bars)) if ratio_bars else None
    return {
        "win_bar_exceedance": {"rank": rank_rate, "ratio": ratio_rate},
        "n_win_bar_windows": {"rank": int(rank_n), "ratio": int(ratio_n)},
        "win_bar_median": {"rank": median_rank, "ratio": median_ratio},
        "win_bar_rank": int(rank_int),
        "win_bar_oracle_ratio": float(ratio),
        "win_bar_is_production_gate": WIN_BAR_IS_PRODUCTION_GATE,
    }
